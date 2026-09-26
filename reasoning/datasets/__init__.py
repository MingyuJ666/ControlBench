"""MATH500 loading, stable IDs and contrastive solution preparation."""
import hashlib
import random
from reasoning.common import DATA, read_jsonl

def problem_id(problem):
    return hashlib.sha256(' '.join(problem.split()).encode()).hexdigest()

def select_subset(rows, limit, seed):
    if not 1 <= limit <= len(rows):
        raise ValueError(f'limit must be between 1 and {len(rows)}')
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [{**rows[i], 'id': problem_id(rows[i]['problem']), 'index': i} for i in indices]

def load(path=DATA/'original.jsonl'):
    return [dict(row, id=problem_id(row['problem']), index=i) for i,row in enumerate(read_jsonl(path))]


def reasoning_prefix(text):
    box = text.rfind('\\boxed{')
    if box < 0:
        return ''
    # Exclude the entire final-answer line, including the gold answer.
    return text[:text.rfind('\n', 0, box) + 1].rstrip()


EXTRACTION_POSITIONS = {
    'before_answer': 'last_non_padding_before_final_answer_line',
    'after_answer': 'last_non_padding_after_full_answer',
}


def extraction_positions(position):
    if position == 'both':
        return tuple(EXTRACTION_POSITIONS)
    if position not in EXTRACTION_POSITIONS:
        raise ValueError(f'Unknown extraction position: {position}')
    return (position,)


def expand_extraction_grid(grid, position):
    """Preserve historical candidates when only one position is requested."""
    positions = extraction_positions(position)
    if position != 'both':
        return grid
    return [dict(candidate, extraction_position=p) for p in positions for candidate in grid]


def add_extraction_argument(parser, *, allow_both=False):
    choices = (*EXTRACTION_POSITIONS, 'both') if allow_both else tuple(EXTRACTION_POSITIONS)
    parser.add_argument('--extraction-position', choices=choices,
                        default='before_answer',
                        help='Capture before the final boxed-answer line (default), '
                             'or at the end of the full answer, including the answer.'
                             + (' Use both to search both positions on dev.' if allow_both else ''))


def extraction_text(text, position='before_answer'):
    """Select completion text; after_answer preserves full-completion extraction."""
    if position == 'before_answer':
        return reasoning_prefix(text)
    if position == 'after_answer':
        return text.rstrip()
    raise ValueError(f'Unknown extraction position: {position}')
