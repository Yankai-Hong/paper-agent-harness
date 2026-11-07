"""Read one query on stdin and return only active model token vectors as JSON."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paper_agent.colqwen_query_encoder import ColQwenQueryEncoder


def main():
    request = json.load(sys.stdin)
    encoder = ColQwenQueryEncoder(request['base'], request['adapter'], request['contract'], device=request['device'])
    vectors = encoder.encode_query(request['query'])
    json.dump({'status': 'ok', 'vectors': vectors.tolist()}, sys.stdout, allow_nan=False)


if __name__ == '__main__':
    main()
