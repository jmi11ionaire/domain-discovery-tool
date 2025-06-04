#!/usr/bin/env python3
"""
Silent Continuous Discovery Runner
Suppresses asyncio warnings for cleaner output during large-scale discovery
"""

import asyncio
import logging
import warnings
import sys
from continuous_discovery_runner import ContinuousDiscoveryRunner

# Suppress asyncio warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up logging to suppress debug messages
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

def exception_handler(loop, context):
    """Handle uncaught asyncio exceptions silently"""
    # Only log actual critical errors, ignore connection issues
    exception = context.get('exception')
    if exception:
        error_type = type(exception).__name__
        if error_type in ['ClientConnectionError', 'ConnectionResetError', 'TimeoutError']:
            # Silently ignore connection issues - they're expected
            return
    
    # Log other exceptions at debug level
    msg = context.get("exception", context["message"])
    logging.debug(f"Asyncio exception: {msg}")

async def main():
    """Run silent continuous discovery"""
    # Set custom exception handler
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(exception_handler)
    
    print("🚀 Silent Continuous Domain Discovery")
    print("Running until 2000 approved domains found")
    print("Connection errors will be handled silently")
    print("Press Ctrl+C to stop gracefully\n")
    
    runner = ContinuousDiscoveryRunner(target_approved=2000)
    await runner.run_continuous_discovery()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Discovery stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
