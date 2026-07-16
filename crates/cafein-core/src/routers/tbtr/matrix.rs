//! The matrix scan state and pass: exact-round arrivals with winner
//! records, canonical elections included.

use super::*;

/// How a stop's exact-round arrival in a matrix pass was achieved.
#[derive(Clone, Copy)]
pub(super) enum StopWinner {
    Unreached,
    /// Reached by the access walk alone (round 0), leaving at
    /// `departure` and walking `seconds` — the chain root the canonical
    /// path key compares.
    Access {
        departure: u32,
        seconds: u32,
    },
    /// Alighted `arena[segment]` at `alight`.
    Alight {
        segment: u32,
        alight: u16,
    },
    /// Walked a footpath from `from` after alighting `arena[segment]`
    /// at `alight`.
    Walked {
        segment: u32,
        alight: u16,
        from: StopIdx,
    },
}

/// Pooled per-worker state for the cost-matrix passes: the profile
/// scan's scratch plus exact-round arrivals and the winner records the
/// cost reconstruction walks. One state serves an origin's descending
/// departures (horizons, labels, arena, and winners persist across
/// passes, rTBTR-style); `reset` recycles it for the next origin.
pub struct MatrixState {
    pub(super) rounds: usize,
    pub(super) reached: Vec<u16>,
    pub(super) arena: Vec<Segment>,
    pub(super) queues: Vec<Vec<(u32, u16)>>,
    /// The at-most-round arrival gate, `stop × rounds` with the suffix
    /// write of the profile scan.
    pub(super) labels: Vec<u32>,
    pub(super) walked: WalkedScratch,
    /// Exact-round arrivals, `stop × (rounds + 1)`; slot 0 is the
    /// access-only round, slot `q + 1` holds queue round `q`'s alights
    /// and their same-round walks — RAPTOR's `tau` shape.
    pub(super) tau: Vec<u32>,
    pub(super) winners: Vec<StopWinner>,
    /// The unscanned queue owner per `(view trip, round)` (an arena
    /// index; `u32::MAX` empty): an equal-boarding segment offered
    /// while the owner is still pending replaces its origin when the
    /// challenger's parent chain is canonical — the pre-label tie site
    /// a horizon rejection would otherwise discard silently.
    pub(super) pending: Vec<u32>,
    /// Reusable scratch for canonical-key comparisons on exact ties.
    pub(super) key_scratch_a: Vec<crate::path_key::PathToken>,
    pub(super) key_scratch_b: Vec<crate::path_key::PathToken>,
}

impl MatrixState {
    /// Fresh state sized for an engine and a round count.
    pub fn new(engine: &TbtrEngine<'_>, max_transfers: u8) -> MatrixState {
        let rounds = max_transfers as usize + 1;
        let stop_count = engine.timetable.stop_count() as usize;
        MatrixState {
            rounds,
            reached: engine.horizons(rounds),
            arena: Vec::new(),
            queues: vec![Vec::new(); rounds],
            labels: vec![UNREACHED; stop_count * rounds],
            walked: WalkedScratch::new(stop_count),
            tau: vec![UNREACHED; stop_count * (rounds + 1)],
            winners: vec![StopWinner::Unreached; stop_count * (rounds + 1)],
            pending: vec![u32::MAX; engine.view.trip_count() as usize * rounds],
            key_scratch_a: Vec::new(),
            key_scratch_b: Vec::new(),
        }
    }

    /// Recycles the buffers for another origin on the same engine.
    pub fn reset(&mut self, engine: &TbtrEngine<'_>) {
        self.reached.clear();
        self.reached
            .extend_from_slice(&engine.horizons(self.rounds));
        self.arena.clear();
        for queue in &mut self.queues {
            queue.clear();
        }
        self.labels.fill(UNREACHED);
        self.walked.clear();
        self.tau.fill(UNREACHED);
        self.winners.fill(StopWinner::Unreached);
        self.pending.fill(u32::MAX);
    }

    pub(super) fn tau_at(&self, stop: StopIdx, round: usize) -> u32 {
        self.tau[stop.0 as usize * (self.rounds + 1) + round]
    }

    pub(super) fn record(&mut self, stop: StopIdx, round: usize, arrival: u32, winner: StopWinner) {
        let slot = stop.0 as usize * (self.rounds + 1) + round;
        self.tau[slot] = arrival;
        self.winners[slot] = winner;
    }

    pub(super) fn winner_at(&self, stop: StopIdx, round: usize) -> StopWinner {
        self.winners[stop.0 as usize * (self.rounds + 1) + round]
    }
}

impl<'a> TbtrEngine<'a> {
    /// One scan of a departure onto a `MatrixState`: the profile scan
    /// stripped of destination targets and journey assembly, writing
    /// exact-round arrivals and winner records instead — with the
    /// canonical path key electing among exact arrival ties at every
    /// site an equal-time alternative could otherwise be discarded
    /// (access seeds, alights, walks, and equal boardings of a pending
    /// queue owner). State persists across strictly decreasing
    /// departures (rTBTR); `reset` recycles it between origins.
    pub fn matrix_pass(&self, departure: u32, access: &[(StopIdx, u32)], state: &mut MatrixState) {
        let rounds = state.rounds;
        for &(stop, seconds) in access {
            let at = departure.saturating_add(seconds);
            improve_labels(&mut state.labels, rounds, stop, at, 0);
            if at < state.tau_at(stop, 0) {
                state.record(stop, 0, at, StopWinner::Access { departure, seconds });
            } else if at == state.tau_at(stop, 0) && at != UNREACHED {
                let mut challenger = std::mem::take(&mut state.key_scratch_a);
                let mut incumbent = std::mem::take(&mut state.key_scratch_b);
                challenger.clear();
                challenger.push(PathToken::Access {
                    stop: stop.0,
                    duration: seconds,
                });
                incumbent.clear();
                let incumbent_root = self.winner_tokens_into(state, stop, 0, &mut incumbent);
                if challenger_wins(departure, &challenger, incumbent_root, &incumbent) {
                    state.record(stop, 0, at, StopWinner::Access { departure, seconds });
                }
                state.key_scratch_a = challenger;
                state.key_scratch_b = incumbent;
            }
        }
        self.seed_matrix(departure, access, state);
        for round in 0..rounds {
            if state.queues[round].is_empty() {
                break;
            }
            let segments = std::mem::take(&mut state.queues[round]);
            // The drained queue is scanned: its owners stop being
            // pending for origin replacement.
            for &(segment, _) in &segments {
                let trip = state.arena[segment as usize].trip;
                state.pending[trip.0 as usize * rounds + round] = u32::MAX;
            }
            state.walked.clear();
            for &(segment, end) in &segments {
                let trip = state.arena[segment as usize].trip;
                let board = state.arena[segment as usize].board;
                let line = self.view.line_of(trip);
                let offset = self.view.line_day_offset(line);
                let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                let times = self.view.stored_times(self.timetable, trip);
                let last = (end as usize + 1).min(times.len()) as u16;
                for alight in board + 1..last {
                    let arrival = times[alight as usize].arrival - offset;
                    let stop = stops[alight as usize];
                    // Walks relax only from arrivals that improve the
                    // stop's at-most-round label — RAPTOR's marked-stop
                    // semantics; the same improvements write the
                    // exact-round winner the cost reconstruction walks.
                    if improve_labels(&mut state.labels, rounds, stop, arrival, round)
                        || (arrival == state.tau_at(stop, round + 1)
                            && arrival != UNREACHED
                            && self.alight_tie_wins(state, stop, round + 1, segment, alight))
                    {
                        state.record(
                            stop,
                            round + 1,
                            arrival,
                            StopWinner::Alight { segment, alight },
                        );
                        self.relax_matrix_walks(state, stop, arrival, segment, alight, round);
                    }
                    if round + 1 < rounds {
                        for transfer in self.set.from_trip_position(trip, alight) {
                            self.enqueue_matrix(
                                state,
                                round + 1,
                                Segment {
                                    trip: transfer.trip,
                                    board: transfer.position,
                                    origin: SegmentOrigin::Transfer {
                                        parent: segment,
                                        alight,
                                    },
                                },
                            );
                        }
                    }
                }
            }
            if round + 1 < rounds {
                let improved: Vec<u32> = state.walked.iter().map(|(stop, _)| stop).collect();
                for stop in improved {
                    // Board from the stop's *final* winner: a canonical
                    // tie replacement may have installed a different
                    // walked chain, or a direct alight — whose boardings
                    // the precomputed transfer set already covers.
                    let StopWinner::Walked {
                        segment, alight, ..
                    } = state.winner_at(StopIdx(stop), round + 1)
                    else {
                        continue;
                    };
                    let ready = state.tau_at(StopIdx(stop), round + 1);
                    self.board_walked_matrix(
                        state,
                        StopIdx(stop),
                        ready,
                        segment,
                        alight,
                        round + 1,
                    );
                }
            }
        }
    }

    /// One footpath hop from a freshly recorded alight, with the same
    /// strict-then-canonical admission as the alight itself.
    fn relax_matrix_walks(
        &self,
        state: &mut MatrixState,
        stop: StopIdx,
        arrival: u32,
        segment: u32,
        alight: u16,
        round: usize,
    ) {
        let rounds = state.rounds;
        for footpath in self.footpaths.from_stop(stop) {
            let walked_at = arrival.saturating_add(footpath.duration);
            if improve_labels(&mut state.labels, rounds, footpath.to, walked_at, round) {
                state.record(
                    footpath.to,
                    round + 1,
                    walked_at,
                    StopWinner::Walked {
                        segment,
                        alight,
                        from: stop,
                    },
                );
                state
                    .walked
                    .insert(footpath.to.0, (walked_at, segment, alight));
            } else if walked_at == state.tau_at(footpath.to, round + 1) && walked_at != UNREACHED {
                let mut challenger = std::mem::take(&mut state.key_scratch_a);
                let mut incumbent = std::mem::take(&mut state.key_scratch_b);
                challenger.clear();
                challenger.push(PathToken::Walk {
                    from: stop.0,
                    to: footpath.to.0,
                    duration: footpath.duration,
                });
                let root = self.segment_tokens_into(state, segment, alight, &mut challenger);
                incumbent.clear();
                let incumbent_root =
                    self.winner_tokens_into(state, footpath.to, round + 1, &mut incumbent);
                let wins = challenger_wins(root, &challenger, incumbent_root, &incumbent);
                state.key_scratch_a = challenger;
                state.key_scratch_b = incumbent;
                if wins {
                    state.record(
                        footpath.to,
                        round + 1,
                        walked_at,
                        StopWinner::Walked {
                            segment,
                            alight,
                            from: stop,
                        },
                    );
                    state
                        .walked
                        .insert(footpath.to.0, (walked_at, segment, alight));
                }
            }
        }
    }

    /// Whether an equal-arrival alight canonically replaces the winner
    /// at `(stop, round_slot)`.
    fn alight_tie_wins(
        &self,
        state: &mut MatrixState,
        stop: StopIdx,
        round_slot: usize,
        segment: u32,
        alight: u16,
    ) -> bool {
        let mut challenger = std::mem::take(&mut state.key_scratch_a);
        let mut incumbent = std::mem::take(&mut state.key_scratch_b);
        challenger.clear();
        let root = self.segment_tokens_into(state, segment, alight, &mut challenger);
        incumbent.clear();
        let incumbent_root = self.winner_tokens_into(state, stop, round_slot, &mut incumbent);
        let wins = challenger_wins(root, &challenger, incumbent_root, &incumbent);
        state.key_scratch_a = challenger;
        state.key_scratch_b = incumbent;
        wins
    }

    /// Seeds round 0 onto the matrix state, electing canonically among
    /// equal boardings.
    fn seed_matrix(&self, departure: u32, access: &[(StopIdx, u32)], state: &mut MatrixState) {
        for &(stop, seconds) in access {
            let ready = departure.saturating_add(seconds);
            for served in self.timetable.patterns_at_stop(stop) {
                for line in self
                    .view
                    .lines_of_pattern(served.pattern)
                    .into_iter()
                    .flatten()
                {
                    let Some(boarded) = earliest_boardable(
                        &self.view,
                        self.timetable,
                        line,
                        served.position,
                        ready,
                    ) else {
                        continue;
                    };
                    self.enqueue_matrix(
                        state,
                        0,
                        Segment {
                            trip: boarded,
                            board: served.position,
                            origin: SegmentOrigin::Access {
                                stop,
                                seconds,
                                departure,
                            },
                        },
                    );
                }
            }
        }
    }

    /// Boards every line catchable at a walked stop, canonically.
    fn board_walked_matrix(
        &self,
        state: &mut MatrixState,
        stop: StopIdx,
        ready: u32,
        parent: u32,
        alight: u16,
        round: usize,
    ) {
        for served in self.timetable.patterns_at_stop(stop) {
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(boarded) =
                    earliest_boardable(&self.view, self.timetable, line, served.position, ready)
                else {
                    continue;
                };
                self.enqueue_matrix(
                    state,
                    round,
                    Segment {
                        trip: boarded,
                        board: served.position,
                        origin: SegmentOrigin::Transfer { parent, alight },
                    },
                );
            }
        }
    }

    /// `enqueue` with the pre-label tie site: an equal boarding of a
    /// still-pending queue owner replaces the owner's origin when the
    /// challenger's parent chain is canonical. A horizon rejection
    /// would otherwise discard the canonical chain before any label
    /// write could compare it.
    fn enqueue_matrix(&self, state: &mut MatrixState, round: usize, segment: Segment) {
        let rounds = state.rounds;
        let trip = segment.trip;
        let board = segment.board;
        let slot = trip.0 as usize * rounds + round;
        if board > state.reached[slot] {
            return;
        }
        if board < state.reached[slot] {
            // A strictly earlier boarding while the owner is still
            // pending replaces it in place, inheriting its scan end: a
            // same-trip later boarding reaches every shared position at
            // the same second and is never electable (RAPTOR boards at
            // the earliest catchable position), so its chain must not
            // keep the positions beyond the merged range to itself.
            let owner = state.pending[slot];
            if owner != u32::MAX {
                state.arena[owner as usize].board = board;
                state.arena[owner as usize].origin = segment.origin;
                let line_end = self.view.line_trips(self.view.line_of(trip)).end;
                for later in trip.0..line_end {
                    let base = later as usize * rounds;
                    for horizon in &mut state.reached[base + round..base + rounds] {
                        *horizon = (*horizon).min(board);
                    }
                }
                return;
            }
        }
        if board == state.reached[slot] {
            let owner = state.pending[slot];
            if owner != u32::MAX && state.arena[owner as usize].board == board {
                let owner_origin = match &state.arena[owner as usize].origin {
                    SegmentOrigin::Access {
                        stop,
                        seconds,
                        departure,
                    } => SegmentOrigin::Access {
                        stop: *stop,
                        seconds: *seconds,
                        departure: *departure,
                    },
                    SegmentOrigin::Transfer { parent, alight } => SegmentOrigin::Transfer {
                        parent: *parent,
                        alight: *alight,
                    },
                };
                // RAPTOR boards from the stop's label — the earliest
                // arrival — so the parent that reaches the boarding
                // first wins outright; the canonical key only breaks an
                // exact ready tie.
                let challenger_ready = self.origin_ready(state, &segment.origin, board, trip);
                let owner_ready = self.origin_ready(state, &owner_origin, board, trip);
                if challenger_ready < owner_ready {
                    state.arena[owner as usize].origin = segment.origin;
                } else if challenger_ready == owner_ready {
                    let mut challenger = std::mem::take(&mut state.key_scratch_a);
                    let mut incumbent = std::mem::take(&mut state.key_scratch_b);
                    challenger.clear();
                    let root = self.origin_tokens_into(
                        state,
                        &segment.origin,
                        board,
                        trip,
                        &mut challenger,
                    );
                    incumbent.clear();
                    let incumbent_root =
                        self.origin_tokens_into(state, &owner_origin, board, trip, &mut incumbent);
                    let wins = challenger_wins(root, &challenger, incumbent_root, &incumbent);
                    state.key_scratch_a = challenger;
                    state.key_scratch_b = incumbent;
                    if wins {
                        state.arena[owner as usize].origin = segment.origin;
                    }
                }
            }
            return;
        }
        state.queues[round].push((state.arena.len() as u32, state.reached[slot]));
        state.pending[slot] = state.arena.len() as u32;
        state.arena.push(segment);
        let line_end = self.view.line_trips(self.view.line_of(trip)).end;
        for later in trip.0..line_end {
            let base = later as usize * rounds;
            for horizon in &mut state.reached[base + round..base + rounds] {
                *horizon = (*horizon).min(board);
            }
        }
    }
}
