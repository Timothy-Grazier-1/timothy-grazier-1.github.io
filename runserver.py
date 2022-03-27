
"""
Run a local sevrer to test changes before push to prod
"""

from web import app
from web.config import PORT, DEBUG

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG
    )