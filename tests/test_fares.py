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
        "departure": board,
        "arrival": board + 300,
        "board_stop": board_stop,
        "alight_stop": alight_stop,
    }


def journey(*legs):
    return {
        "departure": legs[0]["departure"] if legs else 0,
        "arrival": legs[-1]["arrival"] if legs else 0,
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
    # column) and whose stops carry no zone_id: loadable, prices nothing.
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
    assert math.isnan(structure.price(journey(ride("R1", 0, "S1", "S1"))))
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


def test_flat_tables_align_with_the_network(network, hsl):
    seeded = fares.setup_fare_structure(network, base_fare=3.0)
    flat = seeded._flat_tables(network)
    assert len(flat["route_type"]) == len(network.routes) == len(flat["route_fare"])
    count = len(flat["unlimited_transfers"])
    assert len(flat["allow_same_route"]) == count
    assert len(flat["pair_fare"]) == count * count
    assert all(kind < count for kind in flat["route_type"])
    assert flat["transfer_allowance"] == seeded.transfer_time_allowance * 60.0
    zones = hsl._flat_tables(network)
    assert len(zones["stop_zone"]) == len(network.stops)
    assert len(zones["products"]) == len(hsl.fares)
    # The ABCD product covers all four zone bits.
    named = dict(zip(hsl.fares["fare_id"], zones["products"]))
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
