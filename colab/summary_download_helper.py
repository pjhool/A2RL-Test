# -*- coding: utf-8 -*-
"""
Kaggle Summary File Download Helper
Creates download links for summary tar.gz files in Kaggle notebooks.
"""

def create_summary_download_links():
    """
    Create download links for all summary tar.gz files in /kaggle/working.
    
    Usage in Kaggle notebook:
        from summary_download_helper import create_summary_download_links
        create_summary_download_links()
    """
    import os
    import glob
    from IPython.display import display, HTML, FileLink
    
    # Find all summary tar.gz files
    working_dir = '/kaggle/working'
    pattern = os.path.join(working_dir, 'summary_*.tar.gz')
    files = sorted(glob.glob(pattern), reverse=True)  # Most recent first
    
    if not files:
        print("❌ No summary files found in /kaggle/working/")
        print("\nSearching for summary directory...")
        
        summary_dir = os.path.join(working_dir, 'summary')
        if os.path.exists(summary_dir):
            # Calculate directory size
            try:
                import subprocess
                result = subprocess.run(['du', '-sh', summary_dir], 
                                      capture_output=True, text=True, check=True)
                dir_size = result.stdout.split()[0]
                print(f"✓ Found summary directory: {summary_dir}")
                print(f"  Size: {dir_size}")
                print("\nTo create archive, run:")
                print("  !cd /kaggle/working && tar -czf summary_manual.tar.gz summary/")
            except Exception as e:
                print(f"  Directory exists but size check failed: {e}")
        else:
            print("❌ Summary directory not found")
            print("  Training may not have started yet or logs are in a different location")
        return
    
    print("=" * 70)
    print(f"📦 Found {len(files)} summary archive(s)")
    print("=" * 70)
    print()
    
    # Display file information and links
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        file_size_bytes = os.path.getsize(filepath)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Extract timestamp from filename
        timestamp = filename.replace('summary_', '').replace('.tar.gz', '')
        
        print(f"[{i}] {filename}")
        print(f"    Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
        print(f"    Timestamp: {timestamp}")
        print(f"    Path: {filepath}")
        
        # Create download link
        print(f"    Download: ", end='')
        display(FileLink(filepath))
        print()
    
    print("=" * 70)
    print("📥 DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("1. Click the blue link(s) above to download")
    print("2. If link doesn't work:")
    print("   - Save notebook: 'Save Version' → 'Quick Save'")
    print("   - Go to 'Output' tab on the right")
    print("   - Download from there")
    print("=" * 70)


def create_latest_summary_link():
    """
    Create download link for the most recent summary file only.
    
    Usage in Kaggle notebook:
        from summary_download_helper import create_latest_summary_link
        create_latest_summary_link()
    """
    import os
    import glob
    from IPython.display import display, FileLink
    
    working_dir = '/kaggle/working'
    pattern = os.path.join(working_dir, 'summary_*.tar.gz')
    files = sorted(glob.glob(pattern), reverse=True)
    
    if not files:
        print("❌ No summary files found")
        return None
    
    latest_file = files[0]
    filename = os.path.basename(latest_file)
    file_size_mb = os.path.getsize(latest_file) / (1024 * 1024)
    
    print("📦 Latest Summary Archive:")
    print(f"   File: {filename}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Download: ", end='')
    display(FileLink(latest_file))
    
    return latest_file


def list_summary_contents(tar_path=None):
    """
    List contents of a summary tar.gz file without extracting.
    
    Args:
        tar_path: Path to tar.gz file. If None, uses the latest file.
    
    Usage:
        from summary_download_helper import list_summary_contents
        list_summary_contents()
    """
    import os
    import glob
    import subprocess
    
    if tar_path is None:
        # Find latest file
        working_dir = '/kaggle/working'
        pattern = os.path.join(working_dir, 'summary_*.tar.gz')
        files = sorted(glob.glob(pattern), reverse=True)
        
        if not files:
            print("❌ No summary files found")
            return
        
        tar_path = files[0]
    
    if not os.path.exists(tar_path):
        print(f"❌ File not found: {tar_path}")
        return
    
    print(f"📋 Contents of {os.path.basename(tar_path)}:")
    print("=" * 70)
    
    try:
        result = subprocess.run(['tar', '-tzf', tar_path], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        
        # Count files by type
        event_files = [l for l in lines if 'events.out' in l]
        
        print(f"Total items: {len(lines)}")
        print(f"Event files: {len(event_files)}")
        print()
        print("Directory structure:")
        
        # Show first 20 lines
        for line in lines[:20]:
            print(f"  {line}")
        
        if len(lines) > 20:
            print(f"  ... and {len(lines) - 20} more items")
        
        print("=" * 70)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list contents: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def create_summary_html_links():
    """
    Create HTML-formatted download links with better styling.
    
    Usage:
        from summary_download_helper import create_summary_html_links
        create_summary_html_links()
    """
    import os
    import glob
    from IPython.display import display, HTML
    
    working_dir = '/kaggle/working'
    pattern = os.path.join(working_dir, 'summary_*.tar.gz')
    files = sorted(glob.glob(pattern), reverse=True)
    
    if not files:
        print("❌ No summary files found")
        return
    
    html_parts = [
        '<div style="font-family: monospace; padding: 20px; background-color: #f5f5f5; border-radius: 5px;">',
        '<h3 style="color: #333;">📦 Summary Archives</h3>',
        '<table style="width: 100%; border-collapse: collapse;">',
        '<tr style="background-color: #ddd;">',
        '<th style="padding: 10px; text-align: left;">File</th>',
        '<th style="padding: 10px; text-align: left;">Size</th>',
        '<th style="padding: 10px; text-align: left;">Timestamp</th>',
        '<th style="padding: 10px; text-align: left;">Download</th>',
        '</tr>'
    ]
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        timestamp = filename.replace('summary_', '').replace('.tar.gz', '')
        
        # Format timestamp
        if len(timestamp) == 15:  # YYYYMMDD_HHMMSS
            formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        else:
            formatted_time = timestamp
        
        bg_color = '#fff' if i % 2 == 0 else '#f9f9f9'
        
        html_parts.append(f'''
        <tr style="background-color: {bg_color};">
            <td style="padding: 10px; font-weight: bold;">{filename}</td>
            <td style="padding: 10px;">{file_size_mb:.2f} MB</td>
            <td style="padding: 10px;">{formatted_time}</td>
            <td style="padding: 10px;">
                <a href="/kaggle/working/{filename}" 
                   download="{filename}"
                   style="background-color: #4CAF50; color: white; padding: 5px 15px; 
                          text-decoration: none; border-radius: 3px; display: inline-block;">
                    ⬇️ Download
                </a>
            </td>
        </tr>
        ''')
    
    html_parts.append('</table>')
    html_parts.append('<p style="margin-top: 20px; color: #666; font-size: 12px;">')
    html_parts.append('💡 Tip: If download link doesn\'t work, use "Save Version" → "Quick Save" and check the "Output" tab.')
    html_parts.append('</p>')
    html_parts.append('</div>')
    
    display(HTML(''.join(html_parts)))


# Convenience function for quick use
def download_summary():
    """
    Quick function to display download links.
    
    Usage in Kaggle notebook:
        from summary_download_helper import download_summary
        download_summary()
    """
    create_summary_download_links()


if __name__ == "__main__":
    # If run as a script, show all links
    print("Summary Download Helper")
    print()
    create_summary_download_links()
