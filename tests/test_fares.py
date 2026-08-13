"""Fare structures and journey pricing, against r5r's Porto Alegre
structure and the HSL zone fares bundled in the Helsinki feed."""

import math

import pytest

from cafein import fares


def ride(route_id, board, board_stop="X", alight_stop="Y"):
    """A transit leg with just the fields pricing consumes."""
    return {
        "type": "transit",
        "route_id": route_id,
        "departure_s": board,
        "arrival_s": board + 300,
        "board_stop": board_stop,
        "alight_stop": alight_stop,
    }


def journey(*legs):
    return {
        "departure_s": legs[0]["departure_s"] if legs else 0,
        "arrival_s": legs[-1]["arrival_s"] if legs else 0,
        "rides": sum(leg["type"] == "transit" for leg in legs),
        "legs": list(legs),
    }


@pytest.fixture(scope="module")
def poa(fares_poa):
    return fares.load_fare_structure(fares_poa)


def test_loads_the_r5r_structure(poa):
    assert poa.max_discounted_transfers == 1
    assert poa.transfer_time_allowance == 60.0
    assert math.isinf(poa.fare_cap)
    rail = poa.fares_per_type[poa.fares_per_type["type"] == "RAIL"].iloc[0]
    assert bool(rail["unlimited_transfers"])
    assert rail["fare"] == 4.5
    bus_bus = poa.fares_per_transfer[
        (poa.fares_per_transfer["first_leg"] == "BUS")
        & (poa.fares_per_transfer["second_leg"] == "BUS")
    ].iloc[0]
    assert bus_bus["fare"] == 7.2
    # The vignette removed the RAIL-RAIL pair: unlimited transfers cover it.
    assert not (
        (poa.fares_per_transfer["first_leg"] == "RAIL")
        & (poa.fares_per_transfer["second_leg"] == "RAIL")
    ).any()
    assert len(poa.fares_per_route) == 117


def test_prices_follow_the_r5r_vignette(poa):
    # Single legs pay their mode's fare; walking is free.
    assert poa.price(journey(ride("1112", 0))) == pytest.approx(4.8)
    assert poa.price(journey(ride("LINHA1", 0))) == pytest.approx(4.5)
    assert poa.price(journey()) == 0.0
    # Bus to bus within the hour integrates at the pair total of 7.20;
    # a late second boarding pays both full fares.
    assert poa.price(journey(ride("1112", 0), ride("149", 1800))) == pytest.approx(7.2)
    assert poa.price(journey(ride("1112", 0), ride("149", 3601))) == pytest.approx(9.6)
    # Bus and rail integrate at 8.37 either way around.
    assert poa.price(journey(ride("1112", 0), ride("LINHA1", 1800))) == pytest.approx(
        8.37
    )
    assert poa.price(journey(ride("LINHA1", 0), ride("1112", 1800))) == pytest.approx(
        8.37
    )
    # Rail rides after rail are free (unlimited transfers), spending
    # neither the discount nor the transfer clock: a bus after two rails
    # still integrates.
    assert poa.price(
        journey(ride("LINHA1", 0), ride("LINHAAERO", 1800))
    ) == pytest.approx(4.5)
    assert poa.price(
        journey(ride("LINHA1", 0), ride("LINHAAERO", 1800), ride("1112", 3000))
    ) == pytest.approx(8.37)
    # Only one discounted transfer: the third bus pays in full.
    assert poa.price(
        journey(ride("1112", 0), ride("149", 1200), ride("165", 2400))
    ) == pytest.approx(7.2 + 4.8)
    # Reboarding the same bus route is not an integration.
    assert poa.price(journey(ride("1112", 0), ride("1112", 1800))) == pytest.approx(9.6)
    # A route without a fare row cannot be priced.
    assert math.isnan(poa.price(journey(ride("NO_SUCH_ROUTE", 0))))


def test_fare_cap_limits_the_total(poa):
    capped = fares.FareStructure(
        max_discounted_transfers=poa.max_discounted_transfers,
        transfer_time_allowance=poa.transfer_time_allowance,
        fare_cap=8.0,
        fares_per_type=poa.fares_per_type,
        fares_per_transfer=poa.fares_per_transfer,
        fares_per_route=poa.fares_per_route,
    )
    assert capped.price(
        journey(ride("1112", 0), ride("149", 1200), ride("165", 2400))
    ) == pytest.approx(8.0)


def test_structures_round_trip_through_disk(poa, tmp_path):
    path = tmp_path / "fares.zip"
    fares.save_fare_structure(poa, path)
    again = fares.load_fare_structure(path)
    assert again.max_discounted_transfers == poa.max_discounted_transfers
    assert again.transfer_time_allowance == poa.transfer_time_allowance
    assert math.isinf(again.fare_cap)
    assert again.fares_per_type.equals(poa.fares_per_type)
    assert again.fares_per_transfer.equals(poa.fares_per_transfer)
    assert again.fares_per_route["route_id"].equals(poa.fares_per_route["route_id"])


def test_setup_seeds_a_structure_from_the_network(network):
    structure = fares.setup_fare_structure(network, base_fare=3.0)
    kinds = set(structure.fares_per_type["type"])
    assert "BUS" in kinds and "SUBWAY" in kinds and "FERRY" in kinds
    assert len(structure.fares_per_route) == len(network.routes)
    assert (structure.fares_per_type["fare"] == 3.0).all()
    assert len(structure.fares_per_transfer) == len(kinds) ** 2
    # The seeded structure prices every pair at the base fare.
    route_id = structure.fares_per_route["route_id"].iloc[0]
    assert structure.price(journey(ride(route_id, 0))) == 3.0
    generic = fares.setup_fare_structure(network, base_fare=3.0, by="GENERIC")
    assert set(generic.fares_per_type["type"]) == {"GENERIC"}
    with pytest.raises(ValueError, match="MODE"):
        fares.setup_fare_structure(network, base_fare=3.0, by="ZONE")


@pytest.fixture(scope="module")
def hsl_zones(helsinki_gtfs):
    return fares.zone_fare_structure(helsinki_gtfs, rules="zones")


@pytest.fixture(scope="module")
def hsl(helsinki_gtfs):
    return fares.zone_fare_structure(helsinki_gtfs)


def test_zone_structure_reads_the_hsl_feed(hsl):
    assert len(hsl.fares) == 7
    assert hsl.fare_zones["AB"] == frozenset({"A", "B"})
    assert hsl.fare_zones["ABCD"] == frozenset({"A", "B", "C", "D"})
    assert hsl.stop_zones["1040602"] == "A"
    assert hsl.stop_zones["4810551"] == "C"


def test_zone_prices_cover_the_journeys_zones(hsl):
    # Within zone A the cheapest covering product is AB at 2.80.
    inside = journey(ride("any", 0, "1040602", "1040280"))
    assert hsl.price(inside) == pytest.approx(2.8)
    # Korso (C) to Käpylä (A) needs ABC at 4.10.
    across = journey(ride("any", 0, "4810551", "1250551"))
    assert hsl.price(across) == pytest.approx(4.1)
    # Two boardings inside the 80-minute AB window ride on one ticket; a
    # boarding beyond it buys a second one.
    linked = journey(
        ride("any", 0, "1040602", "1040280"),
        ride("any", 1800, "1040280", "1040602"),
    )
    assert hsl.price(linked) == pytest.approx(2.8)
    expired = journey(
        ride("any", 0, "1040602", "1040280"),
        ride("any", 7200, "1040280", "1040602"),
    )
    assert hsl.price(expired) == pytest.approx(5.6)
    # A stop without a zone cannot be priced; walking is free.
    assert math.isnan(hsl.price(journey(ride("any", 0, "nowhere", "1040602"))))
    assert hsl.price(journey()) == 0.0


def test_zone_structure_tolerates_missing_fare_columns(tmp_path):
    import zipfile as zf

    # A feed whose fare rules are route-keyed only (no contains_id
    # column) and whose stops carry no zone_id: the route rows become
    # the route grant, so riding that route prices — and other routes
    # still cannot.
    path = tmp_path / "route_fares.zip"
    with zf.ZipFile(path, "w") as archive:
        archive.writestr(
            "fare_attributes.txt",
            "fare_id,price,currency_type,payment_method,transfers\nF1,2.0,EUR,0,\n",
        )
        archive.writestr("fare_rules.txt", "fare_id,route_id\nF1,R1\n")
        archive.writestr("stops.txt", "stop_id,stop_name\nS1,One\n")
    structure = fares.zone_fare_structure(path)
    assert structure.fare_zones == {}
    assert structure.fare_routes == {"F1": frozenset({"R1"})}
    assert structure.price(journey(ride("R1", 0, "S1", "S1"))) == pytest.approx(2.0)
    assert math.isnan(structure.price(journey(ride("R2", 0, "S1", "S1"))))
    # The pre-grant zone-only reading stays available for the compiled
    # matrix path.
    zones_only = fares.zone_fare_structure(path, rules="zones")
    assert math.isnan(zones_only.price(journey(ride("R1", 0, "S1", "S1"))))
    # Without the optional transfers/transfer_duration columns a zone
    # product is simply valid without limits.
    minimal = tmp_path / "minimal.zip"
    with zf.ZipFile(minimal, "w") as archive:
        archive.writestr(
            "fare_attributes.txt",
            "fare_id,price,currency_type,payment_method\nZ,2.0,EUR,0\n",
        )
        archive.writestr("fare_rules.txt", "fare_id,contains_id\nZ,A\n")
        archive.writestr("stops.txt", "stop_id,stop_name,zone_id\nS1,One,A\n")
    open_ended = fares.zone_fare_structure(minimal)
    long_trip = journey(ride("any", 0, "S1", "S1"), ride("any", 50_000, "S1", "S1"))
    assert open_ended.price(long_trip) == pytest.approx(2.0)
    # A feed without fare files says so.
    bare = tmp_path / "bare.zip"
    with zf.ZipFile(bare, "w") as archive:
        archive.writestr("stops.txt", "stop_id,stop_name\nS1,One\n")
    with pytest.raises(ValueError, match="no GTFS fare files"):
        fares.zone_fare_structure(bare)


def test_flat_tables_align_with_the_network(network, hsl, hsl_zones):
    seeded = fares.setup_fare_structure(network, base_fare=3.0)
    flat = seeded._flat_tables(network)
    assert len(flat["route_type"]) == len(network.routes) == len(flat["route_fare"])
    count = len(flat["unlimited_transfers"])
    assert len(flat["allow_same_route"]) == count
    assert len(flat["pair_fare"]) == count * count
    assert all(kind < count for kind in flat["route_type"])
    assert flat["transfer_allowance"] == seeded.transfer_time_allowance * 60.0
    # The compiled pricer carries the zone model only: a structure
    # with rule grants is rejected, never silently priced differently.
    with pytest.raises(ValueError, match="matrix fare pricing"):
        hsl._flat_tables(network)
    zones = hsl_zones._flat_tables(network)
    assert len(zones["stop_zone"]) == len(network.stops)
    assert len(zones["products"]) == len(hsl_zones.fares)
    # The ABCD product covers all four zone bits.
    named = dict(zip(hsl_zones.fares["fare_id"], zones["products"]))
    assert bin(named["ABCD"][1]).count("1") == 4


def test_frontier_carries_fares(network, hsl):
    from cafein import journey_frontier, least_emissions

    frame = journey_frontier(
        network,
        "4810551",
        "1250551",
        "2022-02-22",
        "08:30:00",
        window=600,
        fares=hsl,
    )
    assert "fare" in frame.columns
    assert len(frame)
    # Every candidate crosses C to A: the ABC ticket prices them all,
    # so the equal fares leave the frontier membership unchanged.
    assert frame["fare"].tolist() == pytest.approx([4.1] * len(frame))
    assert frame["frontier"].any()
    assert least_emissions(frame) is not None


def street_leg(mode, departure, seconds):
    """A rebuilt street leg with just the fields pricing consumes."""
    return {
        "type": "access",
        "mode": mode,
        "departure_s": departure,
        "arrival_s": departure + seconds,
    }


def test_street_rentals_price_beside_the_transit_fare(hsl):
    scooter = fares.ZoneFareStructure(
        hsl.fares,
        hsl.fare_zones,
        hsl.stop_zones,
        street={"e_scooter": {"unlock": 1.0, "per_minute": 0.25}},
    )
    # A 90-second rental bills two started minutes: 1.00 + 2 × 0.25.
    trip = journey(
        street_leg("e_scooter", 0, 90), ride("any", 120, "1040602", "1040280")
    )
    priced = fares.annotate_fares([trip], scooter, shared_modes=("e_scooter",))
    assert priced[0]["fare"] == pytest.approx(2.8 + 1.5)
    # The minute boundary is exact: 60 s is one minute, 61 s is two.
    exact = journey(street_leg("e_scooter", 0, 60))
    over = journey(street_leg("e_scooter", 0, 61))
    fares.annotate_fares([exact, over], scooter, shared_modes=("e_scooter",))
    assert exact["fare"] == pytest.approx(1.25)
    assert over["fare"] == pytest.approx(1.5)
    # Own vehicles and walking are free: a fare is what is paid —
    # walking even when mistakenly marked shared.
    own = journey(
        street_leg("bicycle", 0, 90),
        street_leg("walk", 100, 200),
        ride("any", 400, "1040602", "1040280"),
    )
    fares.annotate_fares([own], scooter, shared_modes=("e_scooter", "walk"))
    assert own["fare"] == pytest.approx(2.8)
    # Without shared_modes the scooter leg is treated as owned.
    unmarked = journey(street_leg("e_scooter", 0, 90))
    fares.annotate_fares([unmarked], scooter)
    assert unmarked["fare"] == 0.0


def test_unpriced_rental_modes_are_nan(hsl):
    # A ridden rental mode without a tariff prices NaN, never zero.
    trip = journey(street_leg("e_bike", 0, 90), ride("any", 120, "1040602", "1040280"))
    fares.annotate_fares([trip], hsl, shared_modes=("e_bike",))
    assert math.isnan(trip["fare"])


def test_rule_structures_price_street_rentals_too(poa):
    priced = fares.FareStructure(
        fares_per_type=poa.fares_per_type,
        fares_per_transfer=poa.fares_per_transfer,
        fares_per_route=poa.fares_per_route,
        street={"e_scooter": {"unlock": 2.0, "per_minute": 0.5}},
    )
    scoot = journey(street_leg("e_scooter", 0, 300))
    fares.annotate_fares([scoot], priced, shared_modes=("e_scooter",))
    assert scoot["fare"] == pytest.approx(2.0 + 5 * 0.5)


def test_negative_street_tariffs_are_rejected(hsl):
    with pytest.raises(ValueError, match="must not be negative"):
        fares.ZoneFareStructure(
            hsl.fares,
            hsl.fare_zones,
            hsl.stop_zones,
            street={"e_scooter": {"unlock": -1.0, "per_minute": 0.25}},
        )
    # Both components are stated explicitly; a partial entry could
    # silently undercharge.
    with pytest.raises(ValueError, match="missing"):
        fares.ZoneFareStructure(
            hsl.fares,
            hsl.fare_zones,
            hsl.stop_zones,
            street={"e_scooter": {"unlock": 1.0}},
        )


def test_rule_grants_price_the_fixtures_restricted_fares(hsl):
    # The loader models every rule shape of the real HSL tables.
    assert hsl.fare_routes["BC"] == frozenset({"2550"})
    assert hsl.fare_routes["BCD"] == frozenset({"2550"})
    assert ("Ei HSL", "D", "9665A") in hsl.fare_od["D"]
    assert hsl.fare_routes["D"] == frozenset({"9987"})
    assert ("D", "Ei HSL", "9665A") in hsl.fare_od["D"]
    # Ticket D still covers a plain D-zone trip beside its route-only
    # row — the dimensions are alternative grants, never conjuncts.
    d_stop = next(s for s, z in hsl.stop_zones.items() if z == "D")
    assert hsl.price(journey(ride("any", 0, d_stop, d_stop))) == pytest.approx(2.8)
    # The 9665A feeder prices D↔"Ei HSL" trips through its OD clause.
    outside = next(s for s, z in hsl.stop_zones.items() if z == "Ei HSL")
    od_trip = journey(ride("9665A", 0, d_stop, outside))
    assert hsl.price(od_trip) == pytest.approx(2.8)
    reverse = journey(ride("9665A", 0, outside, d_stop))
    assert hsl.price(reverse) == pytest.approx(2.8)
    # The same trip off the named route stays unpriceable.
    assert math.isnan(hsl.price(journey(ride("other", 0, d_stop, outside))))
    # Route 9987 rides on ticket D wherever it goes.
    assert hsl.price(journey(ride("9987", 0, "1040602", "1040280"))) == (
        pytest.approx(2.8)
    )
    # Route 2550's tickets: its route grant sells BC (2.80) even where
    # the stop zones are outside every zone set, and the same trip off
    # the route stays unpriceable.
    fringe = [s for s, z in hsl.stop_zones.items() if z == "Ei HSL"]
    trip_2550 = journey(ride("2550", 0, fringe[0], fringe[1]))
    assert hsl.price(trip_2550) == pytest.approx(2.8)
    assert math.isnan(hsl.price(journey(ride("other", 0, fringe[0], fringe[1]))))


def test_route_grants_cover_transfers_and_od_endpoints_are_terminal(tmp_path):
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "fare_id": "R",
                "price": 3.0,
                "currency_type": "EUR",
                "transfers": math.nan,
                "transfer_duration": 3600,
            },
            {
                "fare_id": "OD",
                "price": 1.0,
                "currency_type": "EUR",
                "transfers": math.nan,
                "transfer_duration": 3600,
            },
        ]
    )
    route_only = fares.ZoneFareStructure(
        frame[frame["fare_id"] == "R"],
        {},
        {"S1": "A", "S2": "B", "S3": "C"},
        fare_routes={"R": {"X", "Y"}},
    )
    # One route-granted ticket covers a transfer between two of its
    # routes; a leg outside the set breaks the stretch.
    two_routes = journey(ride("X", 0, "S1", "S2"), ride("Y", 600, "S2", "S3"))
    assert route_only.price(two_routes) == pytest.approx(3.0)
    outside = journey(ride("X", 0, "S1", "S2"), ride("Z", 600, "S2", "S3"))
    assert math.isnan(route_only.price(outside))
    # The OD clause reads the stretch's endpoints, not each leg: A→B→C
    # prices as one A→C ticket though no leg is A→C itself.
    od_only = fares.ZoneFareStructure(
        frame[frame["fare_id"] == "OD"],
        {},
        {"S1": "A", "S2": "B", "S3": "C"},
        fare_od={"OD": [("A", "C", None)]},
    )
    chained = journey(ride("X", 0, "S1", "S2"), ride("Y", 600, "S2", "S3"))
    assert od_only.price(chained) == pytest.approx(1.0)
    # An intermediate endpoint alone never satisfies the clause.
    assert math.isnan(od_only.price(journey(ride("X", 0, "S1", "S2"))))


def test_mixed_rule_rows_and_bare_route_owners_normalize(tmp_path):
    import zipfile as zf

    # A row carrying both contains_id and route_id contributes its
    # zone alone — no route grant, or the fare would turn valid on
    # that route outside its covered zones. And in a single-agency
    # feed whose routes omit agency_id, an agency-scoped fare still
    # resolves to that agency's routes rather than an empty scope.
    path = tmp_path / "mixed.zip"
    with zf.ZipFile(path, "w") as archive:
        archive.writestr(
            "agency.txt",
            "agency_id,agency_name,agency_url,agency_timezone\n"
            "A1,One,https://one,Europe/Helsinki\n",
        )
        archive.writestr(
            "routes.txt",
            "route_id,route_short_name,route_type\nR1,r1,3\n",
        )
        archive.writestr(
            "fare_attributes.txt",
            "fare_id,price,currency_type,payment_method,transfers,agency_id\n"
            "F1,2.0,EUR,0,,A1\n",
        )
        archive.writestr("fare_rules.txt", "fare_id,route_id,contains_id\nF1,R1,Z\n")
        archive.writestr("stops.txt", "stop_id,stop_name,zone_id\nS1,One,Z\nS2,Two,\n")
    structure = fares.zone_fare_structure(path)
    assert structure.fare_routes == {}
    assert structure.fare_zones == {"F1": frozenset({"Z"})}
    assert structure.agency_routes == {"F1": frozenset({"R1"})}
    # A fare with no rule rows at all is unrestricted — the spec's
    # reading of a fare without fare_rules — and prices any stretch.
    flat = tmp_path / "flat.zip"
    import zipfile as zf2

    with zf2.ZipFile(path) as source, zf2.ZipFile(flat, "w") as target:
        for name in source.namelist():
            if name == "fare_attributes.txt":
                target.writestr(
                    name,
                    "fare_id,price,currency_type,payment_method,transfers\n"
                    "ANY,1.5,EUR,0,\n",
                )
            elif name == "fare_rules.txt":
                target.writestr(name, "fare_id,route_id\n")
            else:
                target.writestr(name, source.read(name))
    unrestricted = fares.zone_fare_structure(flat)
    assert unrestricted.price(journey(ride("R9", 0, "S1", "S2"))) == (
        pytest.approx(1.5)
    )
    # The zone grant prices within Z on the agency's route; the same
    # route outside the covered zone stays unpriceable.
    assert structure.price(journey(ride("R1", 0, "S1", "S1"))) == pytest.approx(2.0)
    assert math.isnan(structure.price(journey(ride("R1", 0, "S1", "S2"))))


def test_agency_scope_bounds_every_grant(tmp_path):
    import zipfile as zf

    path = tmp_path / "two_agencies.zip"
    with zf.ZipFile(path, "w") as archive:
        archive.writestr(
            "agency.txt",
            "agency_id,agency_name,agency_url,agency_timezone\n"
            "A1,One,https://one,Europe/Helsinki\n"
            "A2,Two,https://two,Europe/Helsinki\n",
        )
        archive.writestr(
            "routes.txt",
            "route_id,agency_id,route_short_name,route_type\n"
            "R1,A1,r1,3\nR2,A2,r2,3\n",
        )
        archive.writestr(
            "fare_attributes.txt",
            "fare_id,price,currency_type,payment_method,transfers,agency_id\n"
            "F1,2.0,EUR,0,,A1\n",
        )
        archive.writestr("fare_rules.txt", "fare_id,contains_id\nF1,Z\n")
        archive.writestr("stops.txt", "stop_id,stop_name,zone_id\nS1,One,Z\nS2,Two,Z\n")
    structure = fares.zone_fare_structure(path)
    # The zone grant holds, but only on the fare's own agency's routes.
    assert structure.price(journey(ride("R1", 0, "S1", "S2"))) == pytest.approx(2.0)
    assert math.isnan(structure.price(journey(ride("R2", 0, "S1", "S2"))))
    # A multi-agency feed whose fares omit agency_id is rejected.
    unscoped = tmp_path / "unscoped.zip"
    with zf.ZipFile(path) as source, zf.ZipFile(unscoped, "w") as target:
        for name in source.namelist():
            if name == "fare_attributes.txt":
                target.writestr(
                    name,
                    "fare_id,price,currency_type,payment_method,transfers\n"
                    "F1,2.0,EUR,0,\n",
                )
            else:
                target.writestr(name, source.read(name))
    with pytest.raises(ValueError, match="multi-agency"):
        fares.zone_fare_structure(unscoped)
    # The zones-only reading still loads such feeds.
    fares.zone_fare_structure(unscoped, rules="zones")


def test_compiled_pricer_matches_python_over_random_sequences(network):
    import random

    rng = random.Random(0x5EED)
    structure = fares.setup_fare_structure(network, base_fare=3.0)
    # Diversify the tables: random per-route fares, mixed route-fare
    # use, some unlimited-transfer types, random pair totals, a short
    # window — every calculator branch gets exercised.
    structure.fares_per_route["route_fare"] = [
        round(rng.uniform(1.0, 6.0), 2) for _ in range(len(structure.fares_per_route))
    ]
    structure.fares_per_type["use_route_fare"] = [
        rng.random() < 0.5 for _ in range(len(structure.fares_per_type))
    ]
    structure.fares_per_type["unlimited_transfers"] = [
        rng.random() < 0.3 for _ in range(len(structure.fares_per_type))
    ]
    structure.fares_per_transfer["fare"] = [
        round(rng.uniform(2.0, 8.0), 2)
        for _ in range(len(structure.fares_per_transfer))
    ]
    structure.transfer_time_allowance = 45.0
    route_ids = [route_id for route_id, _, _ in network.routes]
    for cap in (math.inf, 7.5):
        structure.fare_cap = cap
        flat = structure._flat_tables(network)
        for _ in range(300):
            time, legs, positions = 0, [], []
            for _ in range(rng.randint(1, 5)):
                time += rng.randint(0, 3000)
                position = rng.randrange(len(route_ids))
                positions.append((position, time))
                legs.append(ride(route_ids[position], time))
            expected = structure.price(journey(*legs))
            probed = network._core._fare_probe(flat, positions)
            assert (math.isnan(expected) and math.isnan(probed)) or (
                expected == pytest.approx(probed)
            )


def test_zone_fare_frontier_routes_the_exact_engine(network, hsl_zones):
    from cafein import journey_frontier
    from cafein.frontier import fare_frontier

    # Two pinned witnesses under the sample feed's 2022 tariff
    # (AB/BC/D 2.80, CD 3.20, ABC/BCD 4.10, ABCD 5.70). The C-to-A
    # pair has NO row under the 2.80 cutoff — the cheapest covering
    # ticket is the 4.10 ABC — and the A-zone metro hop rides an AB
    # single at every cutoff. Exact equality both ways: an underpriced
    # or overpriced engine fails these rows.
    cutoffs = [2.80, 4.10, 5.70]
    frame = fare_frontier(
        network,
        ["4810551", "1040601"],
        ["1250551", "1121601"],
        "2022-02-22",
        "08:30:00",
        600,
        hsl_zones,
        cutoffs=cutoffs,
    )

    def rows(from_id, to_id):
        cell = frame[(frame["from_id"] == from_id) & (frame["to_id"] == to_id)]
        return sorted(
            (row["cutoff"], row["travel_time_s"], row["fare"], row["rides"])
            for _, row in cell.iterrows()
        )

    assert rows("4810551", "1250551") == [
        (4.10, 1320, 4.10, 1),
        (5.70, 1320, 4.10, 1),
    ]
    assert rows("1040601", "1121601") == [
        (2.80, 300, 2.80, 1),
        (4.10, 300, 2.80, 1),
        (5.70, 300, 2.80, 1),
    ]
    # And the fold can never beat the exact engine.
    candidates = journey_frontier(
        network,
        "4810551",
        "1250551",
        "2022-02-22",
        "08:30:00",
        window=600,
        fares=hsl_zones,
    )
    priced = candidates["fare"].dropna()
    if len(priced):
        assert frame["fare"].min() <= priced.min() + 1e-9


def test_zone_fare_frontier_rejects_the_fast_discipline(network, hsl_zones):
    from cafein.frontier import fare_frontier

    with pytest.raises(ValueError, match="always exact"):
        fare_frontier(
            network,
            ["4810551"],
            ["1250551"],
            "2022-02-22",
            "08:30:00",
            600,
            hsl_zones,
            cutoffs=[4.50],
            exact=False,
        )


def test_zone_fare_frontier_names_the_grant_limitation(network, hsl):
    # The default reading carries route/OD grants the compiled pricer
    # rejects; the frontier must say how to proceed, not fail deep in
    # the core.
    from cafein.frontier import fare_frontier

    with pytest.raises(ValueError, match='rules="zones"'):
        fare_frontier(
            network,
            ["4810551"],
            ["1250551"],
            "2022-02-22",
            "08:30:00",
            600,
            hsl,
            cutoffs=[4.10],
        )
