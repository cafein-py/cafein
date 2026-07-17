//! Canonical-token walkers over segment chains, mirroring RAPTOR's
//! label-chain tokens.

use super::*;

impl<'a> TbtrEngine<'a> {
    /// When the chain behind a boarding is ready at its boarding stop:
    /// the parent's alight arrival plus the walk into the stop, or the
    /// access arrival.
    pub(super) fn origin_ready(
        &self,
        state: &MatrixState,
        origin: &SegmentOrigin,
        board: u16,
        trip: ViewTrip,
    ) -> u32 {
        let line = self.view.line_of(trip);
        let board_stop = self.timetable.pattern_stops(self.view.line_pattern(line))[board as usize];
        match origin {
            SegmentOrigin::Access {
                stop,
                seconds,
                departure,
            } => {
                let at = departure.saturating_add(*seconds);
                if *stop != board_stop {
                    at.saturating_add(self.walk_duration(*stop, board_stop))
                } else {
                    at
                }
            }
            SegmentOrigin::Transfer { parent, alight } => {
                let parent_entry = &state.arena[*parent as usize];
                let parent_line = self.view.line_of(parent_entry.trip);
                let parent_stop = self
                    .timetable
                    .pattern_stops(self.view.line_pattern(parent_line))[*alight as usize];
                let times = self.view.stored_times(self.timetable, parent_entry.trip);
                let at = times[*alight as usize].arrival - self.view.line_day_offset(parent_line);
                if parent_stop != board_stop {
                    at.saturating_add(self.walk_duration(parent_stop, board_stop))
                } else {
                    at
                }
            }
        }
    }

    /// Appends the canonical tokens of the chain *behind* a boarding —
    /// the walk into the boarding stop (if any) plus the parent chain —
    /// and returns its root departure.
    pub(super) fn origin_tokens_into(
        &self,
        state: &MatrixState,
        origin: &SegmentOrigin,
        board: u16,
        trip: ViewTrip,
        out: &mut Vec<PathToken>,
    ) -> u32 {
        let line = self.view.line_of(trip);
        let board_stop = self.timetable.pattern_stops(self.view.line_pattern(line))[board as usize];
        match origin {
            SegmentOrigin::Access {
                stop,
                seconds,
                departure,
            } => {
                if *stop != board_stop {
                    out.push(PathToken::Walk {
                        from: stop.0,
                        to: board_stop.0,
                        duration: self.walk_duration(*stop, board_stop),
                    });
                }
                out.push(PathToken::Access {
                    stop: stop.0,
                    duration: *seconds,
                });
                *departure
            }
            SegmentOrigin::Transfer { parent, alight } => {
                let parent_entry = &state.arena[*parent as usize];
                let parent_line = self.view.line_of(parent_entry.trip);
                let parent_stop = self
                    .timetable
                    .pattern_stops(self.view.line_pattern(parent_line))[*alight as usize];
                if parent_stop != board_stop {
                    out.push(PathToken::Walk {
                        from: parent_stop.0,
                        to: board_stop.0,
                        duration: self.walk_duration(parent_stop, board_stop),
                    });
                }
                self.segment_tokens_into(state, *parent, *alight, out)
            }
        }
    }

    /// Appends the canonical tokens of an alighted segment chain,
    /// destination → origin, and returns its root departure.
    pub(super) fn segment_tokens_into(
        &self,
        state: &MatrixState,
        segment: u32,
        alight: u16,
        out: &mut Vec<PathToken>,
    ) -> u32 {
        let mut segment = segment;
        let mut alight = alight;
        loop {
            let entry = &state.arena[segment as usize];
            let backing = self.view.backing(entry.trip);
            out.push(PathToken::Ride {
                trip: backing.0,
                day_offset: self.view.day_offset(entry.trip),
                board: entry.board,
                alight,
            });
            match &entry.origin {
                SegmentOrigin::Access {
                    stop,
                    seconds,
                    departure,
                } => {
                    out.push(PathToken::Access {
                        stop: stop.0,
                        duration: *seconds,
                    });
                    return *departure;
                }
                SegmentOrigin::Transfer {
                    parent,
                    alight: parent_alight,
                } => {
                    let line = self.view.line_of(entry.trip);
                    let board_stop = self.timetable.pattern_stops(self.view.line_pattern(line))
                        [entry.board as usize];
                    let parent_entry = &state.arena[*parent as usize];
                    let parent_line = self.view.line_of(parent_entry.trip);
                    let parent_stop = self
                        .timetable
                        .pattern_stops(self.view.line_pattern(parent_line))
                        [*parent_alight as usize];
                    if parent_stop != board_stop {
                        out.push(PathToken::Walk {
                            from: parent_stop.0,
                            to: board_stop.0,
                            duration: self.walk_duration(parent_stop, board_stop),
                        });
                    }
                    let next = *parent;
                    alight = *parent_alight;
                    segment = next;
                }
            }
        }
    }

    /// Appends the canonical tokens behind a stop's recorded winner and
    /// returns its root departure.
    pub(super) fn winner_tokens_into(
        &self,
        state: &MatrixState,
        stop: StopIdx,
        round_slot: usize,
        out: &mut Vec<PathToken>,
    ) -> u32 {
        match state.winner_at(stop, round_slot) {
            StopWinner::Unreached => {
                unreachable!("canonical key walked an unreached winner")
            }
            StopWinner::Access { departure, seconds } => {
                out.push(PathToken::Access {
                    stop: stop.0,
                    duration: seconds,
                });
                departure
            }
            StopWinner::Alight { segment, alight } => {
                self.segment_tokens_into(state, segment, alight, out)
            }
            StopWinner::Walked {
                segment,
                alight,
                from,
            } => {
                out.push(PathToken::Walk {
                    from: from.0,
                    to: stop.0,
                    duration: self.walk_duration(from, stop),
                });
                self.segment_tokens_into(state, segment, alight, out)
            }
        }
    }

    /// The duration of the (deduplicated) footpath between two stops.
    pub(super) fn walk_duration(&self, from: StopIdx, to: StopIdx) -> u32 {
        self.footpaths
            .from_stop(from)
            .iter()
            .find(|transfer| transfer.to == to)
            .map(|transfer| transfer.duration)
            .unwrap_or(0)
    }
}
