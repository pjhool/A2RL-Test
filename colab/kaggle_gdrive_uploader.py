import os
import json
import logging

# Logger setup (assuming it's configured in the main script, but providing a fallback)
logger = logging.getLogger('A2RL')

# --- Google Drive API Imports ---
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    HAS_GDRIVE_LIBS = True
except ImportError:
    HAS_GDRIVE_LIBS = False


class KaggleGoogleDriveUploader:
    """Integrated Google Drive Uploader for Kaggle"""
    
    def __init__(self, token_path='/kaggle/input/secrets/token.json'):
        """
        Initialize the uploader with the path to the GDrive token.
        
        Args:
            token_path (str): Path to token.json (Kaggle Secrets)
        """
        self.token_path = token_path
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """
        Authenticate using:
        1. Google Colab User Data (google.colab.userdata)
        2. token.json file
        3. Kaggle Secrets
        4. Environment Variable
        """
        try:
            token_data = None
            
            # 1. Try Google Colab User Data (New!)
            try:
                from google.colab import userdata
                try:
                    secret_json = userdata.get('GDRIVE_TOKEN')
                    if secret_json:
                        logger.warning("✓ Found GDRIVE_TOKEN in Colab Secrets")
                        token_data = json.loads(secret_json)
                        logger.warning("✓ Successfully parsed GDrive token JSON")
                except Exception as e:
                    logger.error("  - Colab userdata.get failed or invalid JSON: %s", e)
            except ImportError:
                # Not in Colab environment
                pass

            # 2. Try token.json file
            if not token_data and os.path.exists(self.token_path):
                logger.warning(f"✓ Using GDrive token from file: {self.token_path}")
                try:
                    with open(self.token_path) as f:
                        token_data = json.load(f)
                except Exception as e:
                    logger.error(f"  - Error reading {self.token_path}: {e}")
            
            # 3. Try Kaggle Secrets (UserSecretsClient)
            if not token_data:
                try:
                    from kaggle_secrets import UserSecretsClient
                    user_secrets = UserSecretsClient()
                    secret_json = user_secrets.get_secret("GDRIVE_TOKEN")
                    if secret_json:
                        logger.warning("✓ Using GDrive token from Kaggle Secrets (GDRIVE_TOKEN)")
                        token_data = json.loads(secret_json)
                except Exception:
                    # kaggle_secrets not available or secret not set
                    pass
            
            # 4. Try Environment Variable (A2RL_GDRIVE_TOKEN)
            if not token_data and "A2RL_GDRIVE_TOKEN" in os.environ:
                logger.warning("✓ Using GDrive token from environment variable (A2RL_GDRIVE_TOKEN)")
                token_data = json.loads(os.environ["A2RL_GDRIVE_TOKEN"])
                
            if not token_data:
                logger.error("✗ No Google Drive authentication found.")
                logger.warning("Cloud backup will be disabled. To enable, provide 'GDRIVE_TOKEN' in Colab/Kaggle Secrets.")
                self.service = None
                return
                
            creds = Credentials.from_authorized_user_info(token_data)
            self.service = build('drive', 'v3', credentials=creds)
            logger.warning("✓ Google Drive authentication successful")
            
        except Exception as e:
            logger.error(f"✗ Google Drive authentication failed: {e}")
            self.service = None

    def find_or_create_folder(self, folder_name):
        """Locate an existing folder by name or create a new one."""
        if not self.service:
            logger.error("✗ GDrive service not initialized. Skipping operation.")
            return None
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id)', pageSize=1).execute()
            files = results.get('files', [])
            
            if files:
                return files[0]['id']
            else:
                file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
                folder = self.service.files().create(body=file_metadata, fields='id').execute()
                logger.warning(f"✓ Created new GDrive folder: {folder_name}")
                return folder.get('id')
        except Exception as e:
            logger.error(f"✗ GDrive folder error: {e}")
            return None

    def upload_file(self, file_path, folder_name='A2RL_Results'):
        """Upload a file to a specific GDrive folder."""
        if not self.service:
            logger.error("✗ GDrive service not initialized. Skipping upload.")
            return None
        try:
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"⚠️ Skipping upload: File not found ({file_path})")
                return None
                
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.warning(f"📤 Uploading: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
            
            folder_id = self.find_or_create_folder(folder_name)
            if not folder_id:
                return None
                
            file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id]}
            media = MediaFileUpload(file_path, resumable=True)
            
            request = self.service.files().create(body=file_metadata, media_body=media, fields='id')
            response = None
            
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.debug(f"  Upload progress: {int(status.progress() * 100)}%")
                    
            logger.info(f"✓ Upload complete: {os.path.basename(file_path)}")
            return response.get('id')
            
        except Exception as e:
            logger.error(f"✗ GDrive upload error: {e}")
            return None

    def download_file(self, file_id, local_path):
        """Download a file from GDrive by ID."""
        if not self.service:
            logger.error("✗ GDrive service not initialized. Skipping download.")
            return False
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            logger.info(f"Downloading file ID {file_id} to {local_path}...")
            while done is False:
                status, done = downloader.next_chunk()
                # logger.debug(f"Download {int(status.progress() * 100)}%.")
                
            fh.seek(0)
            with open(local_path, 'wb') as f:
                f.write(fh.read())
            logger.info(f"✓ Downloaded: {local_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            return False

    def download_latest_weights(self, drive_folder_name='save_model', local_dir='./downloaded_weights'):
        """
        Search for the latest model in 'drive_folder_name'.
        ...
        """
        if not self.service:
            logger.error("✗ GDrive service not initialized. Skipping weight download.")
            return None
            
        import tarfile
        
        try:
            # 1. Find root folder ID
            folder_id = self.find_or_create_folder(drive_folder_name)
            if not folder_id:
                logger.error(f"Folder '{drive_folder_name}' not found in Drive.")
                return None
            
            # 2. Get Candidates (Folders and .tar.gz files)
            query = f"'{folder_id}' in parents and (mimeType='application/vnd.google-apps.folder' or name contains '.tar.gz') and trashed=false"
            results = self.service.files().list(
                q=query, 
                orderBy='modifiedTime desc', 
                fields='files(id, name, mimeType, modifiedTime)'
            ).execute()
            candidates = results.get('files', [])
            
            if not candidates:
                logger.warning(f"No models (folders or .tar.gz) found in '{drive_folder_name}'.")
                return None
            
            # Sort by modifiedTime descending (API usually does it, but ensure it)
            #candidates.sort(key=lambda x: x['modifiedTime'], reverse=True)
            
            latest_item = candidates[0]
            logger.info(f"Latests candidate found: {latest_item['name']} ({latest_item['mimeType']})")
            
            final_prefix = None
            
            # --- CASE A: .tar.gz Archive ---
            if 'tar.gz' in latest_item['name']:
                logger.info("Detected compressed model archive.")
                tar_path = os.path.join(local_dir, latest_item['name'])
                
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                    
                if self.download_file(latest_item['id'], tar_path):
                     # Extract
                     logger.info(f"Extracting {tar_path}...")
                     try:
                         with tarfile.open(tar_path, "r:gz") as tar:
                             tar.extractall(path=local_dir)
                         logger.info("✓ Extraction complete.")
                         
                         # Search for actor file in the extracted content
                         # The tar usually contains a folder, so we search recursively in local_dir
                         import glob
                         actor_files = glob.glob(os.path.join(local_dir, "**/*_actor.h5"), recursive=True)
                         if actor_files:
                             # Sort by modification time to get the "latest" inside the tar (if multiple)
                             actor_files.sort(key=os.path.getmtime, reverse=True)
                             latest_actor = actor_files[0]
                             final_prefix = latest_actor.replace('_actor.h5', '')
                             logger.info(f"Found extracted model: {os.path.basename(latest_actor)}")
                         else:
                             logger.error("No *_actor.h5 found in extracted archive.")
                             
                     except Exception as e:
                         logger.error(f"Failed to extract tarball: {e}")
            
            # --- CASE B: Standard Folder ---
            else:
                logger.info("Detected standard model folder.")
                d_id = latest_item['id']
                
                # Find *_actor.h5 files in this date folder
                query_files = f"'{d_id}' in parents and name contains '_actor.h5' and trashed=false"
                file_results = self.service.files().list(
                    q=query_files, 
                    orderBy='modifiedTime desc', 
                    fields='files(id, name, modifiedTime)'
                ).execute()
                actor_files = file_results.get('files', [])
                
                if actor_files:
                    latest_actor = actor_files[0]
                    prefix = latest_actor['name'].replace('_actor.h5', '')
                    
                    if not os.path.exists(local_dir):
                        os.makedirs(local_dir)
                        
                    local_actor_path = os.path.join(local_dir, latest_actor['name'])
                    if self.download_file(latest_actor['id'], local_actor_path):
                        # Attempt to download Critic and Metadata
                        for suffix in ['_critic.h5', '_metadata.json']:
                            target_name = f"{prefix}{suffix}"
                            q_target = f"'{d_id}' in parents and name = '{target_name}' and trashed=false"
                            res_t = self.service.files().list(q=q_target, fields='files(id, name)').execute()
                            files_t = res_t.get('files', [])
                            if files_t:
                                self.download_file(files_t[0]['id'], os.path.join(local_dir, target_name))
                                
                        final_prefix = os.path.join(local_dir, prefix)
            
            return final_prefix

        except Exception as e:
            logger.error(f"✗ GDrive download error: {e}")
            return None


def upload_training_results_to_gdrive(summary_file=None, model_file=None, folder_name='A2RL_Results'):
    """
    Kaggle-specific wrapper to upload training archives.
    
    Args:
        summary_file (str): Path to summary tarball
        model_file (str): Path to model tarball
        folder_name (str): Destination folder in GDrive
        
    Returns:
        bool: True if successful
    """
    if not HAS_GDRIVE_LIBS:
        logger.error("✗ Google Drive libraries not installed.")
        logger.error("TIP: Use !pip install google-api-python-client google-auth-oauthlib")
        return False
        
    try:
        uploader = KaggleGoogleDriveUploader()
        files = [f for f in [summary_file, model_file] if f and os.path.exists(f)]
        
        if not files:
            logger.warning("⚠️ No valid files found for Google Drive upload")
            return False
            
        success_count = 0
        for f in files:
            if uploader.upload_file(f, folder_name=folder_name):
                success_count += 1
                
        return success_count > 0
        
    except Exception as e:
        logger.error(f"✗ Google Drive integration error: {e}")
        return False


if __name__ == "__main__":
    # Configure logging for standalone test
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    token_path = 'Y:\\Project_A2RL\\A2RL-Test\\auth\\token.json' 
    print("\n" + "="*60)
    print("A2RL Kaggle Google Drive Uploader - Standalone Test")
    print("="*60)
    
    # Check for library availability
    if not HAS_GDRIVE_LIBS:
        print("❌ Error: Google Drive libraries not found.")
        print("Run: !pip install google-api-python-client google-auth-oauthlib")
    else:
        try:
            # 1. Test Authentication Sources
            print("Testing authentication sources...")
            
            # Helper to check if any auth is available
            auth_available = os.path.exists(token_path)
            if not auth_available:
                try:
                    from kaggle_secrets import UserSecretsClient
                    if UserSecretsClient().get_secret("GDRIVE_TOKEN"):
                        auth_available = True
                except: pass
            if not auth_available and "A2RL_GDRIVE_TOKEN" in os.environ:
                auth_available = True
                
            if not auth_available:
                print(f"❌ Error: No authentication credentials found.")
                print(f"  - File not found: {token_path}")
                print("  - Kaggle Secret 'GDRIVE_TOKEN' not set.")
                print("  - Env var 'A2RL_GDRIVE_TOKEN' not set.")
                print("\nTo fix this:")
                print("1. Upload token.json to Kaggle Secrets (as a file) OR")
                print("2. Add a Secret string named 'GDRIVE_TOKEN' containing the JSON content.")
            else:
                # 2. Test Initialization and Auth
                print("Initializing uploader...")
                uploader = KaggleGoogleDriveUploader(token_path=token_path)
                
                # 3. Test Folder Access
                print("Checking 'A2RL_Results' folder...")
                folder_id = uploader.find_or_create_folder('A2RL_Results')
                if folder_id:
                    print(f"✓ GDrive Folder ID: {folder_id}")
                
                # 4. Search for files to test upload
                # Look in /kaggle/working for result archives
                working_dir = '/kaggle/working'
                working_dir ='Y:\\Project_A2RL\\A2RL-Test\\auth'
                potential_files = [
                    os.path.join(working_dir, f) 
                    for f in os.listdir(working_dir) 
                    if f.endswith('.tar.gz')
                ]
                
                if potential_files:
                    print(f"\nFound {len(potential_files)} archive(s) in {working_dir}:")
                    for pf in potential_files[:3]: # Show first 3
                        print(f"  - {os.path.basename(pf)}")
                    
                    # Test upload of the first file
                    test_target = potential_files[0]
                    print(f"\n--- Running Test Upload: {os.path.basename(test_target)} ---")
                    uploader.upload_file(test_target)
                else:
                    print("\nℹ️ No .tar.gz files found in /kaggle/working to test upload.")
                    print("To test a real upload, create archives by running the training script first.")
                
                print("\n" + "="*60)
                print("✓ Standalone test completed successfully!")
                print("="*60 + "\n")
                
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            print("Check your token validity and internet connection.")
