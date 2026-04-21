#!/usr/bin/env python3
"""
Interior Mapping Scanner v2.0 - Advanced System Introspection
New Features:
- Real-time monitoring with delta detection
- Capability analysis (CAP_* flags)
- Memory region mapping (VMA analysis)
- Enhanced network state tracking
- Process tree reconstruction
- Anomaly detection
- Timeline tracking
- Performance metrics
"""

import os
import json
import subprocess
import re
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import socket
import struct

class AdvancedSystemScanner:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_id_map = {}
        self.node_counter = 0
        self.scan_history = []
        self.anomalies = []
        self.start_time = time.time()
        
        # Capability flags from kernel headers
        self.capabilities = {
            0: 'CAP_CHOWN', 1: 'CAP_DAC_OVERRIDE', 2: 'CAP_DAC_READ_SEARCH',
            3: 'CAP_FOWNER', 4: 'CAP_FSETID', 5: 'CAP_KILL',
            6: 'CAP_SETGID', 7: 'CAP_SETUID', 8: 'CAP_SETPCAP',
            9: 'CAP_LINUX_IMMUTABLE', 10: 'CAP_NET_BIND_SERVICE',
            11: 'CAP_NET_BROADCAST', 12: 'CAP_NET_ADMIN', 13: 'CAP_NET_RAW',
            14: 'CAP_IPC_LOCK', 15: 'CAP_IPC_OWNER', 16: 'CAP_SYS_MODULE',
            17: 'CAP_SYS_RAWIO', 18: 'CAP_SYS_CHROOT', 19: 'CAP_SYS_PTRACE',
            20: 'CAP_SYS_PACCT', 21: 'CAP_SYS_ADMIN', 22: 'CAP_SYS_BOOT',
            23: 'CAP_SYS_NICE', 24: 'CAP_SYS_RESOURCE', 25: 'CAP_SYS_TIME',
            26: 'CAP_SYS_TTY_CONFIG', 27: 'CAP_MKNOD', 28: 'CAP_LEASE',
            29: 'CAP_AUDIT_WRITE', 30: 'CAP_AUDIT_CONTROL',
            31: 'CAP_SETFCAP', 32: 'CAP_MAC_OVERRIDE', 33: 'CAP_MAC_ADMIN',
            34: 'CAP_SYSLOG', 35: 'CAP_WAKE_ALARM', 36: 'CAP_BLOCK_SUSPEND',
            37: 'CAP_AUDIT_READ'
        }
        
    def get_node_id(self, node_type, identifier):
        """Get or create a unique node ID"""
        key = f"{node_type}:{identifier}"
        if key not in self.node_id_map:
            self.node_id_map[key] = self.node_counter
            self.node_counter += 1
        return self.node_id_map[key]
    
    def add_node(self, node_type, identifier, **attributes):
        """Add a node to the graph"""
        node_id = self.get_node_id(node_type, identifier)
        node = {
            'id': node_id,
            'type': node_type,
            'identifier': str(identifier),
            'scan_time': datetime.now().isoformat(),
            **attributes
        }
        self.nodes.append(node)
        return node_id
    
    def add_edge(self, source_type, source_id, target_type, target_id, relationship, **attributes):
        """Add an edge between nodes"""
        source = self.get_node_id(source_type, source_id)
        target = self.get_node_id(target_type, target_id)
        edge = {
            'source': source,
            'target': target,
            'relationship': relationship,
            **attributes
        }
        self.edges.append(edge)
    
    def parse_capabilities(self, cap_hex):
        """Parse capability bitmask into list of capability names"""
        try:
            cap_value = int(cap_hex, 16)
            caps = []
            for bit, name in self.capabilities.items():
                if cap_value & (1 << bit):
                    caps.append(name)
            return caps
        except:
            return []
    
    def scan_processes_advanced(self):
        """Enhanced process scanning with capabilities, cgroups, and more"""
        print("Scanning processes (advanced)...")
        
        for pid_dir in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(pid_dir.name)
                
                # Read process info
                status_file = pid_dir / 'status'
                cmdline_file = pid_dir / 'cmdline'
                environ_file = pid_dir / 'environ'
                exe_link = pid_dir / 'exe'
                cwd_link = pid_dir / 'cwd'
                limits_file = pid_dir / 'limits'
                
                if not status_file.exists():
                    continue
                
                # Parse status file
                status = {}
                with open(status_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            status[key.strip()] = value.strip()
                
                # Get command line
                cmdline = ''
                if cmdline_file.exists():
                    try:
                        with open(cmdline_file, 'r') as f:
                            cmdline = f.read().replace('\0', ' ').strip()
                    except:
                        pass
                
                # Get environment variables (check for sensitive data)
                env_vars = {}
                sensitive_keys = []
                if environ_file.exists():
                    try:
                        with open(environ_file, 'r') as f:
                            env_data = f.read().split('\0')
                            for item in env_data:
                                if '=' in item:
                                    key, value = item.split('=', 1)
                                    # Check for sensitive keys
                                    if any(s in key.upper() for s in ['PASSWORD', 'TOKEN', 'SECRET', 'KEY', 'API']):
                                        sensitive_keys.append(key)
                    except:
                        pass
                
                # Get executable path
                exe_path = ''
                exe_deleted = False
                try:
                    exe_path = str(exe_link.readlink())
                    if '(deleted)' in exe_path:
                        exe_deleted = True
                except:
                    pass
                
                # Get current working directory
                cwd = ''
                try:
                    cwd = str(cwd_link.readlink())
                except:
                    pass
                
                # Parse capabilities
                cap_effective = self.parse_capabilities(status.get('CapEff', '0'))
                cap_permitted = self.parse_capabilities(status.get('CapPrm', '0'))
                cap_inheritable = self.parse_capabilities(status.get('CapInh', '0'))
                
                # Check for suspicious capabilities
                dangerous_caps = {'CAP_SYS_ADMIN', 'CAP_SYS_PTRACE', 'CAP_NET_ADMIN', 
                                 'CAP_SYS_MODULE', 'CAP_DAC_OVERRIDE'}
                suspicious_caps = [cap for cap in cap_effective if cap in dangerous_caps]
                
                # Get cgroup info
                cgroup_file = pid_dir / 'cgroup'
                cgroups = []
                if cgroup_file.exists():
                    try:
                        with open(cgroup_file, 'r') as f:
                            cgroups = [line.strip() for line in f.readlines()]
                    except:
                        pass
                
                # Detect if in container
                in_container = any('docker' in cg or 'lxc' in cg or 'kubepods' in cg 
                                  for cg in cgroups)
                
                # Calculate process age
                stat_file = pid_dir / 'stat'
                process_age = 0
                if stat_file.exists():
                    try:
                        with open(stat_file, 'r') as f:
                            stat_data = f.read().split()
                            # starttime is at index 21 (in clock ticks)
                            if len(stat_data) > 21:
                                boot_time = self._get_boot_time()
                                start_ticks = int(stat_data[21])
                                process_age = time.time() - (boot_time + start_ticks / os.sysconf(os.sysconf_names['SC_CLK_TCK']))
                    except:
                        pass
                
                # Add process node with enhanced attributes
                self.add_node(
                    'Process',
                    pid,
                    name=status.get('Name', ''),
                    cmdline=cmdline,
                    exe=exe_path,
                    exe_deleted=exe_deleted,
                    cwd=cwd,
                    state=status.get('State', ''),
                    ppid=status.get('PPid', ''),
                    uid=status.get('Uid', '').split()[0] if 'Uid' in status else '',
                    gid=status.get('Gid', '').split()[0] if 'Gid' in status else '',
                    threads=status.get('Threads', ''),
                    vm_size=status.get('VmSize', ''),
                    vm_rss=status.get('VmRSS', ''),
                    vm_peak=status.get('VmPeak', ''),
                    capabilities_effective=cap_effective,
                    capabilities_permitted=cap_permitted,
                    suspicious_caps=suspicious_caps,
                    in_container=in_container,
                    sensitive_env_vars=len(sensitive_keys),
                    process_age_seconds=round(process_age, 2),
                    seccomp=status.get('Seccomp', 'N/A')
                )
                
                # Detect anomalies
                if exe_deleted:
                    self.anomalies.append({
                        'type': 'deleted_executable',
                        'pid': pid,
                        'name': status.get('Name', ''),
                        'severity': 'high',
                        'description': f"Process {pid} running from deleted executable"
                    })
                
                if suspicious_caps and not in_container:
                    self.anomalies.append({
                        'type': 'dangerous_capabilities',
                        'pid': pid,
                        'name': status.get('Name', ''),
                        'capabilities': suspicious_caps,
                        'severity': 'medium',
                        'description': f"Process {pid} has dangerous capabilities: {', '.join(suspicious_caps)}"
                    })
                
                if sensitive_keys:
                    self.anomalies.append({
                        'type': 'sensitive_environment',
                        'pid': pid,
                        'name': status.get('Name', ''),
                        'keys': sensitive_keys,
                        'severity': 'low',
                        'description': f"Process {pid} has {len(sensitive_keys)} sensitive environment variables"
                    })
                
                # Create parent-child relationship
                ppid = status.get('PPid', '')
                if ppid and ppid != '0':
                    self.add_edge('Process', ppid, 'Process', pid, 'PARENT_OF', 
                                 weight=1.0)
                
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    
    def _get_boot_time(self):
        """Get system boot time"""
        try:
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('btime'):
                        return float(line.split()[1])
        except:
            pass
        return time.time()
    
    def scan_memory_regions(self, limit=20):
        """Scan memory regions (VMA) for top processes"""
        print("Scanning memory regions...")
        
        # Get top processes by memory
        process_mem = []
        for pid_dir in Path('/proc').glob('[0-9]*'):
            try:
                status_file = pid_dir / 'status'
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        for line in f:
                            if line.startswith('VmRSS:'):
                                mem_kb = int(line.split()[1])
                                process_mem.append((int(pid_dir.name), mem_kb))
                                break
            except:
                continue
        
        # Sort and take top N
        process_mem.sort(key=lambda x: x[1], reverse=True)
        top_processes = [pid for pid, _ in process_mem[:limit]]
        
        for pid in top_processes:
            maps_file = Path(f'/proc/{pid}/maps')
            if not maps_file.exists():
                continue
            
            try:
                with open(maps_file, 'r') as f:
                    lines = f.readlines()
                
                region_types = defaultdict(int)
                total_size = 0
                executable_regions = 0
                writable_exec = 0
                
                for line in lines:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    addr_range = parts[0]
                    perms = parts[1]
                    path = parts[-1] if len(parts) >= 6 else '[anonymous]'
                    
                    # Calculate size
                    try:
                        start, end = addr_range.split('-')
                        size = int(end, 16) - int(start, 16)
                        total_size += size
                    except:
                        continue
                    
                    # Categorize
                    if '[heap]' in path:
                        region_types['heap'] += 1
                    elif '[stack]' in path:
                        region_types['stack'] += 1
                    elif '[vdso]' in path or '[vvar]' in path:
                        region_types['vdso'] += 1
                    elif path.startswith('/'):
                        region_types['file'] += 1
                    else:
                        region_types['anonymous'] += 1
                    
                    # Check for executable regions
                    if 'x' in perms:
                        executable_regions += 1
                        if 'w' in perms:
                            writable_exec += 1
                
                # Update process node with memory info
                for node in self.nodes:
                    if node['type'] == 'Process' and node['identifier'] == str(pid):
                        node['memory_regions'] = dict(region_types)
                        node['total_mapped_mb'] = round(total_size / (1024 * 1024), 2)
                        node['executable_regions'] = executable_regions
                        node['writable_executable'] = writable_exec
                        break
                
                # Anomaly: writable + executable memory (possible code injection)
                if writable_exec > 0:
                    self.anomalies.append({
                        'type': 'writable_executable_memory',
                        'pid': pid,
                        'count': writable_exec,
                        'severity': 'high',
                        'description': f"Process {pid} has {writable_exec} writable+executable memory regions"
                    })
                    
            except (PermissionError, FileNotFoundError):
                continue
    
    def scan_network_enhanced(self):
        """Enhanced network scanning with connection tracking"""
        print("Scanning network (enhanced)...")
        
        connection_states = defaultdict(int)
        listening_ports = []
        established_connections = []
        
        # Scan TCP
        for tcp_file, family in [(Path('/proc/net/tcp'), 'IPv4'), 
                                  (Path('/proc/net/tcp6'), 'IPv6')]:
            if not tcp_file.exists():
                continue
            
            try:
                with open(tcp_file, 'r') as f:
                    lines = f.readlines()[1:]
                
                for line in lines:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    
                    local_addr = self._parse_address(parts[1], family)
                    remote_addr = self._parse_address(parts[2], family)
                    state_hex = parts[3]
                    inode = parts[9]
                    
                    # Convert hex state to readable
                    state_map = {
                        '01': 'ESTABLISHED', '02': 'SYN_SENT', '03': 'SYN_RECV',
                        '04': 'FIN_WAIT1', '05': 'FIN_WAIT2', '06': 'TIME_WAIT',
                        '07': 'CLOSE', '08': 'CLOSE_WAIT', '09': 'LAST_ACK',
                        '0A': 'LISTEN', '0B': 'CLOSING'
                    }
                    state = state_map.get(state_hex, 'UNKNOWN')
                    connection_states[state] += 1
                    
                    # Find process
                    pid = self._find_socket_owner(inode)
                    
                    if pid:
                        socket_id = f"tcp_{family}_{inode}"
                        self.add_node(
                            'Socket',
                            socket_id,
                            protocol='TCP',
                            local_addr=local_addr,
                            remote_addr=remote_addr,
                            state=state,
                            family=family,
                            inode=inode
                        )
                        
                        self.add_edge('Process', pid, 'Socket', socket_id, 'HAS_SOCKET')
                        
                        # Track interesting connections
                        if state == 'LISTEN':
                            port = local_addr.split(':')[-1]
                            listening_ports.append({
                                'pid': pid,
                                'port': port,
                                'family': family
                            })
                        elif state == 'ESTABLISHED':
                            established_connections.append({
                                'pid': pid,
                                'local': local_addr,
                                'remote': remote_addr
                            })
            except Exception as e:
                print(f"Error scanning TCP ({family}): {e}")
        
        # Store network statistics
        self.network_stats = {
            'connection_states': dict(connection_states),
            'listening_ports': listening_ports,
            'established_count': len(established_connections)
        }
    
    def _parse_address(self, addr_str, family):
        """Parse hex address from /proc/net files"""
        try:
            addr, port = addr_str.split(':')
            port = int(port, 16)
            
            if family == 'IPv4':
                addr_int = int(addr, 16)
                ip = socket.inet_ntoa(struct.pack('<L', addr_int))
            else:
                # IPv6 - simplified
                ip = addr
            
            return f"{ip}:{port}"
        except:
            return addr_str
    
    def _find_socket_owner(self, inode):
        """Find which process owns a socket by inode"""
        for pid_dir in Path('/proc').glob('[0-9]*'):
            try:
                fd_dir = pid_dir / 'fd'
                if not fd_dir.exists():
                    continue
                
                for fd_link in fd_dir.iterdir():
                    try:
                        target = str(fd_link.readlink())
                        if f'socket:[{inode}]' in target:
                            return int(pid_dir.name)
                    except:
                        continue
            except:
                continue
        return None
    
    def scan_file_descriptors(self):
        """Scan open file descriptors with enhanced analysis"""
        print("Scanning file descriptors...")
        
        for pid_dir in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(pid_dir.name)
                fd_dir = pid_dir / 'fd'
                
                if not fd_dir.exists():
                    continue
                
                fd_count = 0
                file_types = defaultdict(int)
                interesting_files = []
                
                for fd_link in fd_dir.iterdir():
                    try:
                        target = str(fd_link.readlink())
                        fd_count += 1
                        
                        # Categorize and analyze
                        if target.startswith('/'):
                            if '/dev/' in target:
                                file_types['device'] += 1
                            elif '/tmp/' in target or '/var/tmp/' in target:
                                file_types['temp'] += 1
                                interesting_files.append(target)
                            elif '/proc/' in target or '/sys/' in target:
                                file_types['procfs'] += 1
                            else:
                                file_types['regular'] += 1
                                # Track interesting files
                                if any(ext in target for ext in ['.log', '.db', '.sqlite', '.sock']):
                                    interesting_files.append(target)
                        elif 'pipe:' in target:
                            file_types['pipe'] += 1
                        elif 'socket:' in target:
                            file_types['socket'] += 1
                        elif 'anon_inode:' in target:
                            file_types['anon'] += 1
                        elif target.startswith('['):
                            file_types['special'] += 1
                            
                    except:
                        continue
                
                # Update process with FD info
                for node in self.nodes:
                    if node['type'] == 'Process' and node['identifier'] == str(pid):
                        node['fd_count'] = fd_count
                        node['fd_types'] = dict(file_types)
                        node['interesting_files'] = interesting_files[:5]  # Limit
                        break
                
                # Anomaly: excessive file descriptors
                if fd_count > 1000:
                    self.anomalies.append({
                        'type': 'excessive_fds',
                        'pid': pid,
                        'count': fd_count,
                        'severity': 'medium',
                        'description': f"Process {pid} has {fd_count} open file descriptors"
                    })
                        
            except:
                continue
    
    def scan_namespaces(self):
        """Scan process namespaces"""
        print("Scanning namespaces...")
        
        namespace_types = ['mnt', 'uts', 'ipc', 'pid', 'net', 'user', 'cgroup']
        namespace_map = defaultdict(list)
        
        for pid_dir in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(pid_dir.name)
                ns_dir = pid_dir / 'ns'
                
                if not ns_dir.exists():
                    continue
                
                for ns_type in namespace_types:
                    ns_file = ns_dir / ns_type
                    if ns_file.exists():
                        try:
                            ns_link = str(ns_file.readlink())
                            match = re.search(r'\[(\d+)\]', ns_link)
                            if match:
                                ns_id = match.group(1)
                                key = f"{ns_type}:{ns_id}"
                                namespace_map[key].append(pid)
                        except:
                            continue
            except:
                continue
        
        # Create namespace nodes
        for ns_key, pids in namespace_map.items():
            if len(pids) > 1:
                ns_type, ns_id = ns_key.split(':')
                
                self.add_node(
                    'Namespace',
                    ns_key,
                    ns_type=ns_type,
                    ns_id=ns_id,
                    process_count=len(pids),
                    pids=pids[:10]  # Limit for display
                )
                
                for pid in pids:
                    self.add_edge('Process', pid, 'Namespace', ns_key, 'IN_NAMESPACE')
    
    def calculate_metrics(self):
        """Calculate system-wide metrics"""
        print("Calculating metrics...")
        
        metrics = {
            'scan_duration': round(time.time() - self.start_time, 2),
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {},
            'anomaly_count': len(self.anomalies),
            'network_stats': getattr(self, 'network_stats', {}),
            'top_memory_processes': [],
            'top_fd_processes': [],
            'process_states': defaultdict(int)
        }
        
        # Count node types
        for node in self.nodes:
            node_type = node['type']
            metrics['node_types'][node_type] = metrics['node_types'].get(node_type, 0) + 1
            
            # Track process states
            if node_type == 'Process':
                state = node.get('state', 'Unknown')
                metrics['process_states'][state] += 1
        
        # Top processes by memory
        process_nodes = [n for n in self.nodes if n['type'] == 'Process']
        process_nodes_mem = [(n['identifier'], n.get('vm_rss', '0 kB')) 
                              for n in process_nodes if 'vm_rss' in n]
        def safe_mem(val):
            try:
                return int(str(val).split()[0])
            except Exception:
                return 0

        process_nodes_mem.sort(key=lambda x: safe_mem(x[1]), reverse=True)
        metrics['top_memory_processes'] = process_nodes_mem[:5]
        
        # Top processes by FD count
        process_nodes_fd = [(n['identifier'], n.get('fd_count', 0)) 
                            for n in process_nodes if 'fd_count' in n]
        process_nodes_fd.sort(key=lambda x: x[1], reverse=True)
        metrics['top_fd_processes'] = process_nodes_fd[:5]
        
        return metrics
    
    def export_graph(self, output_file='graph_data.json'):
        """Export enhanced graph with metrics and anomalies"""
        metrics = self.calculate_metrics()
        
        graph_data = {
            'version': '2.0',
            'scan_timestamp': datetime.now().isoformat(),
            'nodes': self.nodes,
            'links': self.edges,
            'metrics': metrics,
            'anomalies': self.anomalies,
            'metadata': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'node_types': list(set(n['type'] for n in self.nodes)),
                'scan_duration_seconds': metrics['scan_duration']
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Graph exported to {output_file}")
        print(f"{'='*60}")
        print(f"Nodes: {len(self.nodes)}")
        print(f"Edges: {len(self.edges)}")
        print(f"Node types: {metrics['node_types']}")
        print(f"Anomalies detected: {len(self.anomalies)}")
        print(f"Scan duration: {metrics['scan_duration']}s")
        print(f"{'='*60}\n")
        
        return graph_data

def main():
    print("="*60)
    print("Interior Mapping Scanner v2.0 - Advanced Edition")
    print("="*60)
    print()
    
    scanner = AdvancedSystemScanner()
    
    # Run all scans
    scanner.scan_processes_advanced()
    scanner.scan_memory_regions(limit=20)
    scanner.scan_network_enhanced()
    scanner.scan_file_descriptors()
    scanner.scan_namespaces()
    
    # Export
    scanner.export_graph('/home/kali/interiorscan_v2/interiorscan_v2/frontend/graph_data.json')
    
    # Show anomalies
    if scanner.anomalies:
        print("\n⚠️  ANOMALIES DETECTED:")
        print("="*60)
        for i, anomaly in enumerate(scanner.anomalies[:10], 1):
            print(f"{i}. [{anomaly['severity'].upper()}] {anomaly['description']}")
        if len(scanner.anomalies) > 10:
            print(f"... and {len(scanner.anomalies) - 10} more")
        print("="*60)
    
    print("\n✅ Scan complete!")

if __name__ == '__main__':
    main()
