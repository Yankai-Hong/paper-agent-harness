"""User-provided images are inputs, never citation evidence."""
import hashlib
from pathlib import Path


def validate_query_image(path, expected_sha256=None):
    from PIL import Image, UnidentifiedImageError
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
        raise ValueError('Query image must be a local PNG or JPEG file')
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Query image exceeds 20 MiB')
    data = path.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError('Query image changed during the request')
    try:
        with Image.open(path) as image:
            if image.format not in {'PNG', 'JPEG'} or image.width < 1 or image.height < 1 or image.width * image.height > 40_000_000:
                raise ValueError('Unsupported query image format or dimensions')
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid query image') from exc
    return {'input_type':'user_image', 'path':str(path), 'sha256':checksum}
