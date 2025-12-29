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
        Authenticate using token.json, Kaggle Secrets, or environment variables.
        """
        try:
            token_data = None
            
            # 1. Try token.json file
            if os.path.exists(self.token_path):
                logger.warning(f"✓ Using GDrive token from file: {self.token_path}")
                with open(self.token_path) as f:
                    token_data = json.load(f)
            
            # 2. Try Kaggle Secrets (UserSecretsClient)
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
            
            # 3. Try Environment Variable (A2RL_GDRIVE_TOKEN)
            if not token_data and "A2RL_GDRIVE_TOKEN" in os.environ:
                logger.warning("✓ Using GDrive token from environment variable (A2RL_GDRIVE_TOKEN)")
                token_data = json.loads(os.environ["A2RL_GDRIVE_TOKEN"])
                
            if not token_data:
                err_msg = (
                    "✗ No Google Drive authentication found.\n"
                    "Please provide one of the following:\n"
                    f"1. A file at {self.token_path}\n"
                    "2. A Kaggle Secret named 'GDRIVE_TOKEN' (JSON string)\n"
                    "3. An environment variable 'A2RL_GDRIVE_TOKEN' (JSON string)"
                )
                logger.error(err_msg)
                raise FileNotFoundError("Google Drive authentication credentials not found")
                
            creds = Credentials.from_authorized_user_info(token_data)
            self.service = build('drive', 'v3', credentials=creds)
            logger.warning("✓ Google Drive authentication successful")
            
        except Exception as e:
            logger.error(f"✗ Google Drive authentication failed: {e}")
            raise

    def find_or_create_folder(self, folder_name):
        """Locate an existing folder by name or create a new one."""
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
                    
            logger.warning(f"✓ Upload complete: {os.path.basename(file_path)}")
            return response.get('id')
            
        except Exception as e:
            logger.error(f"✗ GDrive upload error: {e}")
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
