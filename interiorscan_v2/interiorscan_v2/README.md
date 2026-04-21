# Interior Mapping Scanner v2.0

**Advanced System Introspection & Anomaly Detection**

A sophisticated real-time system mapping tool that scans, visualizes, and analyzes your system's internal state with anomaly detection, capability analysis, and advanced security monitoring.

## 🆕 What's New in v2.0

### Major Features Added

✨ **Anomaly Detection Engine**
- Detects deleted executables (ghost processes)
- Identifies dangerous capabilities (CAP_SYS_ADMIN, CAP_SYS_PTRACE, etc.)
- Flags excessive file descriptors
- Detects writable+executable memory regions
- Monitors sensitive environment variables

🔍 **Advanced Process Analysis**
- Capability parsing (all 38 Linux capabilities)
- Memory region mapping (VMA analysis)
- Container detection (Docker, LXC, Kubernetes)
- Process age calculation
- Seccomp mode detection
- Environment variable scanning

🌐 **Enhanced Network Monitoring**
- Connection state tracking
- Listening port detection
- Established connection monitoring
- Enhanced socket-to-process mapping

💾 **Memory Analysis**
- Virtual memory area (VMA) enumeration
- Heap/stack/anonymous region tracking
- Executable region detection
- Writable+executable memory flagging
- Memory-mapped file tracking

🔄 **Real-Time Monitoring**
- Auto-refresh capability with configurable intervals
- Live anomaly detection
- Delta change tracking
- Continuous system monitoring

📊 **Advanced Metrics Dashboard**
- Process state distribution
- Network connection statistics
- Top memory consumers
- Top FD consumers
- Scan performance metrics

🔎 **Search & Navigation**
- Real-time node search
- Filter by PID, name, path
- Click-to-focus camera navigation
- Advanced filtering options

## 🎯 Quick Start

### Run the Scanner

```bash
cd backend
python3 scanner_v2.py
```

### Launch Visualization

```bash
cd ../frontend
python3 -m http.server 8000
```

Then open: **http://localhost:8000**

## 📊 Features in Detail

### Anomaly Detection

The scanner automatically detects security-relevant anomalies:

| Anomaly Type | Severity | Description |
|--------------|----------|-------------|
| Deleted Executable | High | Process running from deleted binary (possible rootkit) |
| Writable+Executable Memory | High | Memory regions that can be written AND executed (code injection vector) |
| Dangerous Capabilities | Medium | Processes with CAP_SYS_ADMIN, CAP_SYS_PTRACE, etc. |
| Excessive FDs | Medium | Processes with >1000 open file descriptors |
| Sensitive Environment | Low | Environment variables containing passwords/tokens |

### Capability Analysis

Tracks all 38 Linux capabilities including:
- **CAP_SYS_ADMIN**: Full system administration
- **CAP_SYS_PTRACE**: Trace arbitrary processes
- **CAP_NET_ADMIN**: Network administration
- **CAP_SYS_MODULE**: Load kernel modules
- **CAP_DAC_OVERRIDE**: Bypass file permissions

And 33 more...

### Memory Region Mapping

For top processes by memory usage, scans:
- Heap regions
- Stack regions  
- Anonymous mappings
- File-backed mappings
- Virtual DSO (vdso/vvar)
- Executable regions
- Writable+executable regions (security risk)

### Real-Time Monitoring

Enable auto-refresh to continuously monitor your system:
- Configurable refresh interval (5-300 seconds)
- Visual indicator when active
- Automatic anomaly re-detection
- Live graph updates

### Search Functionality

Quickly find nodes:
- Search by PID
- Search by process name
- Search by file path
- Search by any attribute
- Click result to focus camera on node

## 🎨 Visualization Features

### Color Coding

- **Cyan**: Processes
- **Red**: Sockets / Nodes with anomalies
- **Teal**: Files
- **Yellow**: Namespaces

### Interactive Controls

- **Click** nodes to see detailed information
- **Drag** to rotate the view
- **Scroll** to zoom in/out
- **Right-click + drag** to pan
- **Search** to find and focus nodes

### Panels

1. **Controls Panel** (Top Left)
   - Node type filtering
   - Connection count filtering
   - Auto-refresh settings
   - Search functionality
   - Legend & statistics

2. **Anomalies Panel** (Top Right)
   - Real-time anomaly list
   - Severity indicators
   - Detailed descriptions
   - Count badges

3. **Metrics Panel** (Bottom Left)
   - System-wide metrics
   - Process state distribution
   - Network statistics
   - Performance data

4. **Info Panel** (Bottom Right)
   - Selected node details
   - All attributes displayed
   - Suspicious capability highlighting
   - Relationship information

## 📁 Data Collected

### Process Information

**Basic Attributes:**
- PID, PPID, name, command line
- Executable path, working directory
- Process state, UID, GID
- Thread count
- Process age (uptime)

**Memory:**
- Virtual memory size (VmSize)
- Resident set size (VmRSS)
- Peak memory usage (VmPeak)
- Memory region breakdown
- Executable regions count

**Security:**
- Effective capabilities
- Permitted capabilities
- Inheritable capabilities
- Suspicious capability detection
- Seccomp mode
- Container detection

**Files:**
- File descriptor count
- FD type breakdown (regular, device, pipe, socket)
- Interesting files (logs, databases, sockets)

**Environment:**
- Sensitive environment variable detection
- Count of potentially dangerous env vars

### Network Connections

- Protocol (TCP/UDP)
- Local address:port
- Remote address:port
- Connection state
- Socket inode
- Owning process

### File Descriptors

- FD number
- Path/target
- Type (regular, device, pipe, socket, anon)
- Statistics per process

### Namespaces

- All 7 namespace types:
  - mnt (Mount)
  - uts (Hostname)
  - ipc (Inter-Process Communication)
  - pid (Process ID)
  - net (Network)
  - user (User/UID mapping)
  - cgroup (Control Groups)
- Shared namespace detection
- Process-to-namespace mapping

## 🔒 Security & Privacy

- ✅ **100% Local**: All data stays on your machine
- ✅ **Read-Only**: Scanner only reads from /proc
- ✅ **No Network**: No external connections
- ✅ **Permission-Aware**: Handles permission errors gracefully
- ✅ **Anomaly Detection**: Identifies suspicious behavior
- ✅ **Capability Aware**: Tracks privileged operations

## ⚡ Performance

- **Scan Time**: 2-10 seconds (typical desktop)
- **Node Count**: 500-2000 nodes (typical system)
- **Memory Usage**: ~50-200MB for visualization
- **Refresh**: Supports continuous monitoring
- **Visualization**: Handles 10,000+ nodes smoothly

## 🛠️ Advanced Usage

### Running as Root

For complete system visibility:
```bash
sudo python3 backend/scanner_v2.py
```

### Custom Scan Intervals

Edit auto-refresh in the UI or modify scanner:
```python
# In scanner_v2.py
# Adjust which scans to run
scanner.scan_processes_advanced()
scanner.scan_memory_regions(limit=50)  # Increase limit
scanner.scan_network_enhanced()
scanner.scan_file_descriptors()
scanner.scan_namespaces()
```

### Export Data

The scanner exports to JSON with full metadata:
```json
{
  "version": "2.0",
  "scan_timestamp": "2026-04-20T...",
  "nodes": [...],
  "links": [...],
  "metrics": {...},
  "anomalies": [...]
}
```

### Anomaly Filtering

Filter anomalies by severity in the UI or programmatically:
```javascript
const highSeverity = graphData.anomalies.filter(a => a.severity === 'high');
const mediumSeverity = graphData.anomalies.filter(a => a.severity === 'medium');
```

## 📈 Metrics Explained

### Process States

- **R**: Running
- **S**: Sleeping (interruptible)
- **D**: Sleeping (uninterruptible)
- **Z**: Zombie
- **T**: Stopped
- **t**: Tracing stop
- **I**: Idle

### Connection States

- **ESTABLISHED**: Active connection
- **LISTEN**: Listening for connections
- **SYN_SENT/RECV**: Connection being established
- **TIME_WAIT**: Connection recently closed
- **CLOSE_WAIT**: Waiting for local close
- **FIN_WAIT1/2**: Connection terminating

## 🔧 Troubleshooting

### "Permission denied" Errors

**Solution**: Run with sudo:
```bash
sudo python3 backend/scanner_v2.py
```

### High Memory Usage

**Solution**: Reduce memory region scan limit:
```python
scanner.scan_memory_regions(limit=10)  # Default: 20
```

### Slow Scans

**Solution**: Disable memory region scanning for speed:
```python
# Comment out in scanner_v2.py
# scanner.scan_memory_regions()
```

### Graph Performance Issues

**Solution**: Increase minimum connections filter to reduce visible nodes, or filter by type.

## 🆚 v1.0 vs v2.0 Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Process Scanning | ✅ Basic | ✅ Advanced + Capabilities |
| Network Analysis | ✅ Basic | ✅ Enhanced + Stats |
| Memory Mapping | ❌ | ✅ VMA Analysis |
| Anomaly Detection | ❌ | ✅ 5 Types |
| Capability Detection | ❌ | ✅ All 38 Caps |
| Real-time Monitoring | ❌ | ✅ Auto-refresh |
| Search | ❌ | ✅ Full-text |
| Metrics Dashboard | ❌ | ✅ Comprehensive |
| Container Detection | ❌ | ✅ Docker/K8s/LXC |
| Security Analysis | ❌ | ✅ Multi-layer |

## 🚀 Future Roadmap

Planned enhancements:
- [ ] eBPF integration for kernel-level tracing
- [ ] Neo4j backend for massive graphs
- [ ] Historical timeline view
- [ ] Machine learning anomaly detection
- [ ] Export to SIEM systems
- [ ] Differential scanning (show changes)
- [ ] Process ancestry tree view
- [ ] Binary hash verification
- [ ] Yara rule integration
- [ ] Custom alerting rules

## 📚 Technical Details

### Architecture

```
/proc filesystem
    ↓
Scanner (Python)
    ↓
JSON Graph Data
    ↓
3D Visualization (React + 3d-force-graph)
    ↓
Interactive Analysis
```

### Data Model

**Nodes**: Process, Socket, File, Namespace  
**Edges**: PARENT_OF, HAS_SOCKET, HAS_FD, IN_NAMESPACE  
**Attributes**: 50+ per node type  
**Metrics**: System-wide statistics  
**Anomalies**: Security event list

### Technologies

- **Backend**: Python 3 (stdlib only)
- **Frontend**: React 18 + 3d-force-graph
- **Rendering**: WebGL (Three.js)
- **Layout**: Force-directed (d3-force-3d)
- **Data**: JSON graph format

## 📄 License

Educational and research use.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New anomaly detection rules
- Additional data sources
- Performance optimizations
- Export format support
- Documentation improvements

---

**Built for security research, system analysis, and forensics** 🔍
