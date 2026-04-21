#!/bin/bash

# Interior Mapping Scanner v2.0 - Quick Launch Script
# Advanced system introspection with anomaly detection

echo "=========================================="
echo "Interior Mapping Scanner v2.0"
echo "Advanced Edition with Anomaly Detection"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🔍 Step 1: Running advanced system scan..."
echo ""
echo "Scanning for:"
echo "  • Processes (with capabilities)"
echo "  • Memory regions (VMA analysis)"
echo "  • Network connections (enhanced)"
echo "  • File descriptors"
echo "  • Namespaces"
echo "  • Anomalies"
echo ""

# Run the scanner
cd backend
python3 scanner_v2.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Scanner failed. Try running with sudo for full access:"
    echo "   sudo ./run.sh"
    echo ""
    echo "Without sudo, some processes may not be accessible."
    exit 1
fi

echo ""
echo "✅ Scan complete!"
echo ""
echo "🌐 Step 2: Starting visualization server..."
echo ""

# Start the web server
cd ../frontend
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 Server running at:"
echo "     http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Features:"
echo "   ✓ 3D interactive graph visualization"
echo "   ✓ Real-time anomaly detection"
echo "   ✓ Advanced security analysis"
echo "   ✓ Auto-refresh monitoring"
echo "   ✓ Full-text search"
echo "   ✓ Metrics dashboard"
echo ""
echo "⌨️  Controls:"
echo "   • Click nodes to inspect details"
echo "   • Drag to rotate view"
echo "   • Scroll to zoom"
echo "   • Enable auto-refresh for live monitoring"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m http.server 8000
