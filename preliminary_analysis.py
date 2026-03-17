import chess
import requests, csv, io
from dataclasses import dataclass, field

# ── Piece encoding helper ────────────────────────────────────────────────────

PIECE_VAL = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}

def board_to_vector(board: chess.Board) -> list[int]:
    vec = [0] * 64
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            vec[sq] = PIECE_VAL[p.piece_type] * (1 if p.color == chess.WHITE else -1)
    return vec

def pawn_mask(board: chess.Board) -> tuple[int, int]:
    return (
        int(board.pieces(chess.PAWN, chess.WHITE)),
        int(board.pieces(chess.PAWN, chess.BLACK)),
    )

def fen_to_epd(fen: str) -> str:
    return " ".join(fen.split()[:4])

# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class OpeningResult:
    eco: str
    name: str
    confidence: float       # 0.0 – 1.0
    method: str             # which level matched
    candidates: list = field(default_factory=list)  # top-3 when uncertain

# ── Classifier ───────────────────────────────────────────────────────────────

class OpeningClassifier:

    # A few canonical examples — extend this as needed
    # Pawn structure → opening family mapping
    # NEED TO EDIT THE FAMILIES
    PAWN_FAMILIES = [
        # Sicilian: e4 played, c5 responded — black c-pawn on c5, white e-pawn on e4
        (0x0000000000001000, 0x0000040000000000, "Sicilian Defence"),
        # French: e4 e6 d4 d5
        (0x0000000000081000, 0x0000180000000000, "French Defence"),
        # Caro-Kann: e4 c6 d4 d5
        (0x0000000000081000, 0x0000280000000000, "Caro-Kann Defence"),
        # Queen's Gambit: d4 d5 c4
        (0x0000000000001400, 0x0000080000000000, "Queen's Gambit"),
        # King's Indian: d4 Nf6 c4 g6
        (0x0000000000001400, 0x0000000000000000, "King's Indian / Grünfeld family"),
    ]

    def __init__(self):
        print("Loading ECO database...")
        rows = []
        for url in [
            "https://raw.githubusercontent.com/lichess-org/chess-openings/master/a.tsv",
            "https://raw.githubusercontent.com/lichess-org/chess-openings/master/b.tsv",
            "https://raw.githubusercontent.com/lichess-org/chess-openings/master/c.tsv",
            "https://raw.githubusercontent.com/lichess-org/chess-openings/master/d.tsv",
            "https://raw.githubusercontent.com/lichess-org/chess-openings/master/e.tsv",
        ]:
            rows.extend(csv.DictReader(io.StringIO(requests.get(url).text), delimiter="\t"))

        # Pre-compute vectors for all ECO positions
        self.epd_map = {}       # epd -> row
        self.eco_vectors = []   # list of (vector, pawn_mask, row)

        for row in rows:
            epd = row.get("epd", "").strip()
            if not epd:
                continue
            try:
                board = chess.Board(epd)
                vec = board_to_vector(board)
                pmask = pawn_mask(board)
                self.epd_map[epd] = row
                self.eco_vectors.append((vec, pmask, row))
            except Exception:
                continue

        print(f"Loaded {len(self.eco_vectors)} ECO positions.")

    # ── exact match ─────────────────────────────────────────────────

    def _exact(self, epd: str) -> OpeningResult | None:
        row = self.epd_map.get(epd)
        if row:
            return OpeningResult(row["eco"], row["name"], 1.0, "exact")
        return None

    # ── piece-square similarity ────────────────────────────────────

    def _similarity(self, vec: list[int]) -> list[tuple[float, dict]]:
        """Return top matches sorted by similarity score descending."""
        scored = []
        for eco_vec, _, row in self.eco_vectors:
            matches = sum(a == b for a, b in zip(vec, eco_vec))
            score = matches / 64.0
            scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        return scored[:10]

    # ── Level 3: pawn structure fingerprint ──────────────────────────────────

    def _pawn_family(self, pmask: tuple[int, int]) -> str | None:
        """
        Count pawn overlap with canonical family masks.
        Uses popcount of (input & canonical) / popcount(canonical).
        """
        w_mask, b_mask = pmask
        best_score, best_family = 0.0, None

        for canon_w, canon_b, family in self.PAWN_FAMILIES:
            w_overlap = bin(w_mask & canon_w).count("1")
            b_overlap = bin(b_mask & canon_b).count("1")
            canon_total = bin(canon_w).count("1") + bin(canon_b).count("1")
            if canon_total == 0:
                continue
            score = (w_overlap + b_overlap) / canon_total
            if score > best_score:
                best_score, best_family = score, family

        return best_family if best_score >= 0.6 else None

    # ── Level 4: piece activity heuristics ───────────────────────────────────

    def _activity_heuristics(self, board: chess.Board) -> dict:
        """
        Returns a dict of heuristic signals useful for open/closed
        position classification even when the opening is unknown.
        """
        centre = chess.SquareSet([chess.E4, chess.D4, chess.E5, chess.D5])
        extended = chess.SquareSet([
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.F4, chess.C5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6,
        ])

        white_centre = len(board.pieces(chess.PAWN, chess.WHITE) & centre)
        black_centre = len(board.pieces(chess.PAWN, chess.BLACK) & centre)

        # Development: non-pawn, non-king pieces off their home rank
        white_home = chess.SquareSet(chess.BB_RANK_1)
        black_home = chess.SquareSet(chess.BB_RANK_8)
        white_dev = sum(
            1 for pt in [chess.KNIGHT, chess.BISHOP]
            for sq in board.pieces(pt, chess.WHITE)
            if sq not in white_home
        )
        black_dev = sum(
            1 for pt in [chess.KNIGHT, chess.BISHOP]
            for sq in board.pieces(pt, chess.BLACK)
            if sq not in black_home
        )

        # Open files (no pawns of either color)
        open_files = sum(
            1 for f in range(8)
            if not (board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK))
            & chess.SquareSet(chess.BB_FILES[f])
        )

        # Castling availability
        castled_estimate = (
            not board.has_castling_rights(chess.WHITE),
            not board.has_castling_rights(chess.BLACK),
        )

        return {
            "white_centre_pawns": white_centre,
            "black_centre_pawns": black_centre,
            "white_development": white_dev,
            "black_development": black_dev,
            "open_files": open_files,
            "white_likely_castled": castled_estimate[0],
            "black_likely_castled": castled_estimate[1],
            "position_type": "open" if open_files >= 2 else "semi-open" if open_files == 1 else "closed",
        }

    # ── Main classify method ─────────────────────────────────────────────────

    def classify(self, fen: str, similarity_threshold: float = 0.80) -> OpeningResult:
        epd = fen_to_epd(fen)

        try:
            board = chess.Board(epd)
        except ValueError as e:
            raise ValueError(f"Invalid FEN: {e}")

        vec = board_to_vector(board)
        pmask = pawn_mask(board)

        # ── Level 1
        result = self._exact(epd)
        if result:
            return result

        # ── Level 2
        top = self._similarity(vec)
        best_score, best_row = top[0]

        if best_score >= similarity_threshold:
            candidates = [
                {"eco": r["eco"], "name": r["name"], "score": round(s, 3)}
                for s, r in top[:3]
            ]
            return OpeningResult(
                eco=best_row["eco"],
                name=best_row["name"],
                confidence=round(best_score, 3),
                method="similarity",
                candidates=candidates,
            )

        # ── Level 3: pawn structure
        family = self._pawn_family(pmask)
        if family:
            candidates = [
                {"eco": r["eco"], "name": r["name"], "score": round(s, 3)}
                for s, r in top[:3]
            ]
            return OpeningResult(
                eco="???",
                name=f"{family} (by pawn structure)",
                confidence=0.5,
                method="pawn_structure",
                candidates=candidates,
            )

        # ── Level 4: activity heuristics — enrich the result even if unknown
        hints = self._activity_heuristics(board)
        pos_type = hints["position_type"]
        dev = hints["white_development"] + hints["black_development"]
        guess_family = (
            "Open game (1.e4 e5 family)" if hints["white_centre_pawns"] >= 1 and hints["black_centre_pawns"] >= 1
            else "Semi-open game" if hints["open_files"] == 1
            else "Closed / flank opening"
        )

        candidates = [
            {"eco": r["eco"], "name": r["name"], "score": round(s, 3)}
            for s, r in top[:3]
        ]

        return OpeningResult(
            eco="???",
            name=f"Unknown — likely {guess_family}",
            confidence=0.3,
            method="heuristic",
            candidates=candidates,
        )