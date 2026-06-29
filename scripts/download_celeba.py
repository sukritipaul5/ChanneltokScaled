#!/usr/bin/env python3

"""
CelebA Dataset Download & Extract Script
Handles download, extraction, and preparation for training
"""

import os
import sys
import subprocess
import shutil
import tarfile
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

def run_command(cmd, check=True, shell=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def install_7z():
    """Install 7z if not available"""
    if shutil.which('7z'):
        return True
    
    print("Installing p7zip-full...")
    try:
        if shutil.which('sudo'):
            run_command("sudo apt update && sudo apt install -y p7zip-full")
        else:
            run_command("apt update && apt install -y p7zip-full")
        return True
    except:
        print("Failed to install 7z. Please install manually.")
        return False

def extract_7z(archive_path, target_dir):
    """Extract 7z archive"""
    print(f"Extracting {archive_path}...")
    
    # Check if this is a split archive (has numbered extension)
    is_split_archive = any(archive_path.name.endswith(f'.{i:03d}') for i in range(1, 100))
    
    # For split archives, skip py7zr and go directly to command line 7z
    if not is_split_archive:
        try:
            # Try with py7zr first (pure Python) for single archives
            import py7zr
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                archive.extractall(path=target_dir)
            return True
        except ImportError:
            pass  # Fall through to command line 7z
        except Exception as e:
            print(f"py7zr failed: {e}")
            pass  # Fall through to command line 7z
    
    # Fall back to command line 7z
    if not install_7z():
        return False
    
    try:
        # Change to the directory containing the archive for split archives
        original_dir = Path.cwd()
        archive_dir = archive_path.parent
        archive_name = archive_path.name
        
        # Calculate the relative path from archive directory to target directory
        try:
            # Make both paths absolute to avoid relative path issues
            abs_archive_dir = archive_dir.resolve()
            abs_target_dir = target_dir.resolve()
            
            os.chdir(abs_archive_dir)
            
            # Calculate relative path from archive dir to target dir
            rel_target = os.path.relpath(abs_target_dir, abs_archive_dir)
            
            result = run_command(f'7z x -y "{archive_name}" -o"{rel_target}"', check=False)
            os.chdir(original_dir)
            
            if result.returncode == 0:
                return True
            else:
                print(f"7z extraction failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                
        except Exception as path_error:
            print(f"Path calculation failed: {path_error}")
            os.chdir(original_dir)
            
        # Try simpler extraction without output directory (extract to current dir)
        try:
            os.chdir(archive_dir)
            result = run_command(f'7z x -y "{archive_name}"', check=False)
            os.chdir(original_dir)
            
            if result.returncode == 0:
                return True
            else:
                print(f"Simple extraction failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                
        except Exception as e2:
            print(f"Simple extraction failed with exception: {e2}")
            os.chdir(original_dir)
            
        return False
            
    except Exception as e:
        print(f"Extraction failed with exception: {e}")
        try:
            os.chdir(original_dir)
        except:
            pass
        return False

def count_images(directory):
    """Count image files in directory"""
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count

def create_tarball(source_dir, output_file):
    """Create a tar.gz file from directory"""
    print(f"Creating compressed archive: {output_file}")
    with tarfile.open(output_file, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    
    # Get file size
    size = os.path.getsize(output_file)
    size_mb = size / (1024 * 1024)
    if size_mb > 1024:
        size_str = f"{size_mb/1024:.1f}GB"
    else:
        size_str = f"{size_mb:.1f}MB"
    
    print(f"✓ Created {output_file} ({size_str})")

def download_with_progress(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def main():
    print("CelebA Dataset Setup Script")
    print("===========================")
    print()
    
    # Setup directory
    download_dir = Path("celeba_download")
    download_dir.mkdir(exist_ok=True)
    os.chdir(download_dir)
    
    celeba_dir = Path("CelebA")
    img_dir = celeba_dir / "Img"
    
    # Check if CelebA folder exists with 7z files
    if celeba_dir.exists():
        print("✓ Found CelebA folder")
        
        # Check for 7z files (including split archives in directories)
        main_archive = None
        archive_info = []
        
        # Check for regular 7z files first
        png_7z = img_dir / "img_align_celeba_png.7z"
        celeba_7z = img_dir / "img_celeba.7z"
        
        if png_7z.exists() and png_7z.is_file():
            main_archive = png_7z
            archive_info.append(f"  {png_7z.name} ({png_7z.stat().st_size / (1024 * 1024):.1f}MB)")
        elif celeba_7z.exists() and celeba_7z.is_file():
            main_archive = celeba_7z
            archive_info.append(f"  {celeba_7z.name} ({celeba_7z.stat().st_size / (1024 * 1024):.1f}MB)")
        
        # Check for split archives in directories (unusual structure)
        if not main_archive:
            for potential_dir in img_dir.glob("*.7z"):
                if potential_dir.is_dir():
                    # Look for split archive files inside the directory
                    split_files = sorted(list(potential_dir.glob("*.7z.*")))
                    if split_files:
                        # Find the first part (could be .001, .011, etc.)
                        first_part = split_files[0]
                        main_archive = first_part
                        
                        # Add all parts to info
                        total_size = 0
                        for part in split_files:
                            size_mb = part.stat().st_size / (1024 * 1024)
                            total_size += size_mb
                            archive_info.append(f"  {part.name} ({size_mb:.1f}MB)")
                        
                        archive_info.insert(0, f"Split archive total: {total_size:.1f}MB")
                        break
        
        # Also check for direct split files in img_dir
        if not main_archive:
            split_files = sorted(list(img_dir.glob("*.7z.*")))
            if split_files:
                main_archive = split_files[0]
                for part in split_files:
                    size_mb = part.stat().st_size / (1024 * 1024)
                    archive_info.append(f"  {part.name} ({size_mb:.1f}MB)")
        
        if main_archive:
            print("✓ Found 7z archives:")
            for info in archive_info:
                print(info)
            print()
            
            # Check if already extracted
            extracted_dir = Path("img_align_celeba")
            if extracted_dir.exists():
                image_count = count_images(extracted_dir)
                if image_count > 0:
                    print(f"✓ Dataset already extracted: {image_count} images found")
                    print(f"✓ Ready for training at: {extracted_dir.absolute()}")
                    print()
                    
                    # Ask if user wants to re-extract or proceed
                    print("Options:")
                    print("1) Use existing extracted dataset")
                    print("2) Re-extract from archives (will delete existing)")
                    print("3) Create tar.gz for Backblaze")
                    print("4) Exit")
                    
                    choice = input("Select option (1-4): ").strip()
                    
                    if choice == '1':
                        print("Using existing dataset.")
                        return
                    elif choice == '2':
                        print("Re-extracting dataset...")
                        shutil.rmtree(extracted_dir)
                        # Continue to extraction below
                    elif choice == '3':
                        tarball_path = Path("celeba_images.tar.gz")
                        if not tarball_path.exists():
                            create_tarball(extracted_dir, tarball_path)
                        else:
                            print(f"✓ tar.gz already exists: {tarball_path}")
                        return
                    elif choice == '4':
                        print("Exiting...")
                        return
                    else:
                        print("Invalid option, using existing dataset.")
                        return
            
            # Extract aligned faces (preferred for training)
            if main_archive:
                print(f"Extracting aligned face images from {main_archive.name}...")
                
                if not extract_7z(main_archive, img_dir):
                    print("✗ Extraction failed")
                    sys.exit(1)
                
                # Find extracted directory and move to standard location
                possible_dirs = [
                    img_dir / "img_align_celeba_png",
                    img_dir / "img_align_celeba"
                ]
                
                source_dir = None
                for dir_path in possible_dirs:
                    if dir_path.exists():
                        source_dir = dir_path
                        break
                
                if source_dir:
                    if extracted_dir.exists():
                        shutil.rmtree(extracted_dir)
                    shutil.move(str(source_dir), str(extracted_dir))
                    
                    # Verify extraction
                    image_count = count_images(extracted_dir)
                    if image_count > 0:
                        print(f"✓ Extraction complete: {image_count} images")
                        print(f"✓ Dataset ready at: {extracted_dir.absolute()}")
                    else:
                        print("✗ Extraction failed - no images found")
                        sys.exit(1)
                else:
                    print("✗ Could not find extracted directory")
                    sys.exit(1)
            
            # Offer to create tar.gz
            tarball_path = Path("celeba_images.tar.gz")
            if extracted_dir.exists() and not tarball_path.exists():
                print()
                response = input("Create tar.gz for Backblaze upload? (y/n) ").strip().lower()
                if response == 'y':
                    create_tarball(extracted_dir, tarball_path)
            
            return
    
    # No CelebA folder found - proceed with download options
    print("No existing CelebA data found. Choose download method:")
    print()
    print("1) Download from Kaggle (requires account)")
    print("2) Download via torrent (fastest)")
    print("3) Download from Google Drive")
    print("4) Exit")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == '1':
        print("Kaggle Download Instructions:")
        print("=============================")
        print("1. Go to: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        print("2. Download archive.zip")
        print("3. Extract and find img_align_celeba.zip inside")
        print(f"4. Place in: {Path.cwd()}")
        print("5. Run this script again")
        
    elif choice == '2':
        print("Torrent Download")
        print("================")
        
        # Check for aria2c
        if not shutil.which('aria2c'):
            print("Installing aria2c...")
            try:
                if shutil.which('sudo'):
                    run_command("sudo apt update && sudo apt install -y aria2")
                else:
                    run_command("apt update && apt install -y aria2")
            except:
                print("Failed to install aria2c")
                return
        
        print("Downloading CelebA via torrent...")
        magnet_url = "magnet:?xt=urn:btih:7979c4735621d84c86b1097ad87b5c14f22968a4&dn=celeba_hq.tar"
        
        try:
            run_command(f'aria2c --seed-time=0 --max-connection-per-server=10 --split=10 "{magnet_url}"')
            
            if Path("celeba_hq.tar").exists():
                print("Extracting...")
                with tarfile.open("celeba_hq.tar", 'r') as tar:
                    tar.extractall()
                
                if Path("CelebA-HQ-img").exists():
                    if Path("img_align_celeba").exists():
                        shutil.rmtree("img_align_celeba")
                    shutil.move("CelebA-HQ-img", "img_align_celeba")
                    print(f"✓ Dataset ready at: {Path('img_align_celeba').absolute()}")
        except:
            print("Torrent download failed")
            
    elif choice == '3':
        print("Google Drive Download")
        print("====================")
        
        # Check for gdown
        try:
            import gdown
        except ImportError:
            print("Installing gdown...")
            run_command("pip install gdown")
            import gdown
        
        print("Downloading from Google Drive...")
        try:
            gdown.download(id='0B7EVK8r0v71pZjFTYXZWM3FlRnM', output='img_align_celeba.zip')
            
            if Path("img_align_celeba.zip").exists():
                print("Extracting...")
                with zipfile.ZipFile("img_align_celeba.zip", 'r') as zip_ref:
                    zip_ref.extractall()
                print(f"✓ Dataset ready at: {Path('img_align_celeba').absolute()}")
        except:
            print("Google Drive download failed")
            
    elif choice == '4':
        print("Exiting...")
        return
        
    else:
        print("Invalid option")
        sys.exit(1)
    
    # Final check and tar.gz creation
    extracted_dir = Path("img_align_celeba")
    if extracted_dir.exists():
        image_count = count_images(extracted_dir)
        if image_count > 0:
            print()
            print(f"✓ Dataset ready: {image_count} images")
            print(f"✓ Location: {extracted_dir.absolute()}")
            
            tarball_path = Path("celeba_images.tar.gz")
            if not tarball_path.exists():
                print()
                response = input("Create tar.gz for Backblaze upload? (y/n) ").strip().lower()
                if response == 'y':
                    create_tarball(extracted_dir, tarball_path)
                    print()
                    print("To upload to Backblaze:")
                    print("b2 upload-file celeb-a celeba_images.tar.gz celeba_images.tar.gz")
    
    print()
    print("Script complete!")

if __name__ == "__main__":
    main()
