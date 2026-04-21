# QUICK START GUIDE - v2.0

## 🚀 Get Started in 3 Steps

### Step 1: Extract the Files
Unzip this archive to a folder on your system.

### Step 2: Run the Scanner

**Linux/Mac:**
```bash
./run.sh
```

**Windows:**
```
run.bat
```

**Manual (any OS):**
```bash
cd backend
python3 scanner_v2.py
cd ../frontend
python3 -m http.server 8000
```

### Step 3: Open in Browser
Go to: **http://localhost:8000**

## 🆕 What's New in v2.0?

### Anomaly Detection
The scanner now automatically detects:
- 🔴 **Ghost processes** (deleted executables)
- 🔴 **Dangerous memory** (writable + executable)
- 🟠 **Dangerous capabilities** (CAP_SYS_ADMIN, etc.)
- 🟠 **Excessive file descriptors** (>1000 FDs)
- 🟡 **Sensitive environment vars** (passwords, tokens)

Check the **Anomalies Panel** (top right) for security alerts!

### Real-Time Monitoring
Enable **Auto-refresh** in the Controls panel to:
- Monitor your system continuously
- Detect changes in real-time
- Track new processes and connections
- Update anomaly detection

### Advanced Features
- 🔍 **Search**: Find nodes by PID, name, path
- 📊 **Metrics**: System-wide statistics dashboard
- 🎯 **Capabilities**: See which processes have dangerous permissions
- 💾 **Memory Analysis**: VMA region mapping for top processes
- 🌐 **Network Stats**: Connection state tracking

## 🎮 Controls

### Navigation
- **Click** nodes → See detailed information
- **Drag** → Rotate the 3D view
- **Scroll** → Zoom in/out
- **Right-click + drag** → Pan the view

### Panels
1. **Controls** (Top Left) - Filters, search, auto-refresh
2. **Anomalies** (Top Right) - Security alerts
3. **Metrics** (Bottom Left) - System statistics
4. **Info** (Bottom Right) - Selected node details

### Filters
- **Node Type**: Show only processes, sockets, files, or namespaces
- **Min Connections**: Hide nodes with few connections
- **Auto-refresh**: Monitor system continuously (5-300 sec intervals)

### Search
1. Click "Search" tab in Controls panel
2. Type to search: PID, name, path, any attribute
3. Click result to focus camera on that node

## 🔒 Security Features

### Anomaly Severity Levels
- 🔴 **High**: Immediate attention (ghost processes, code injection vectors)
- 🟠 **Medium**: Review recommended (dangerous caps, high FD counts)
- 🟡 **Low**: Informational (sensitive env vars)

### What Gets Scanned
✓ All processes with full details  
✓ Linux capabilities (38 types)  
✓ Memory regions (heap, stack, executable)  
✓ Network connections (TCP/UDP)  
✓ File descriptors and open files  
✓ Container namespaces  
✓ Environment variables  

### Privacy
- ✅ **100% Local** - All data stays on your machine
- ✅ **Read-Only** - Scanner doesn't modify anything
- ✅ **No Network** - No external connections

## 📊 Understanding the Metrics

### Process States
- **R**: Running or runnable
- **S**: Sleeping (waiting for event)
- **D**: Uninterruptible sleep
- **Z**: Zombie (terminated but not reaped)
- **T**: Stopped (by signal or debugger)

### Connection States
- **ESTABLISHED**: Active connection
- **LISTEN**: Waiting for incoming connections
- **TIME_WAIT**: Connection recently closed

### Capabilities (Examples)
- **CAP_SYS_ADMIN**: Can do almost anything
- **CAP_NET_ADMIN**: Network administration
- **CAP_SYS_PTRACE**: Can trace/debug any process
- **CAP_DAC_OVERRIDE**: Bypass file permissions

## ⚡ Tips & Tricks

### For Better Performance
1. Increase "Min Connections" to reduce visible nodes
2. Filter by node type to focus on specific data
3. Disable auto-refresh when not actively monitoring

### For Security Analysis
1. Check Anomalies panel first
2. Look for red nodes (anomaly detected)
3. Inspect processes with suspicious capabilities
4. Review deleted executables (potential rootkits)
5. Monitor writable+executable memory

### For System Monitoring
1. Enable auto-refresh (30-60 sec interval)
2. Watch for new anomalies
3. Track connection state changes
4. Monitor top memory consumers

## 🐛 Troubleshooting

**"Failed to load graph data"**  
→ Run the scanner first: `python3 backend/scanner_v2.py`

**Permission errors**  
→ Run with admin/root: `sudo ./run.sh` or run as administrator

**Graph not showing**  
→ Access via http://localhost:8000, not by opening HTML file directly

**Too many nodes / slow performance**  
→ Increase "Min Connections" filter or filter by node type

**Auto-refresh not working**  
→ Scanner must complete before visualization refreshes. Increase interval if needed.

## 📚 Learn More

- **Full Documentation**: See README.md
- **Changelog**: See CHANGELOG.md for all v2.0 features
- **Technical Spec**: See docs/SPECIFICATION.txt

## 🆚 v1.0 vs v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Anomaly Detection | ❌ | ✅ |
| Capability Analysis | ❌ | ✅ |
| Memory Mapping | ❌ | ✅ |
| Real-Time Monitoring | ❌ | ✅ |
| Search | ❌ | ✅ |
| Metrics Dashboard | ❌ | ✅ |

## 💡 Example Workflows

### Security Audit
1. Run scanner with sudo for full access
2. Check Anomalies panel for high-severity alerts
3. Inspect processes with CAP_SYS_ADMIN
4. Look for writable+executable memory
5. Review deleted executables

### System Monitoring
1. Enable auto-refresh (60 sec)
2. Watch Metrics panel for trends
3. Monitor connection states
4. Track new processes appearing
5. Check for FD leaks (excessive FDs)

### Troubleshooting
1. Search for specific process by name
2. Click to see all details
3. Check file descriptors
4. Review network connections
5. Inspect memory usage

---

**Questions?** Check the full README.md for comprehensive documentation!

**Happy hunting!** 🔍
