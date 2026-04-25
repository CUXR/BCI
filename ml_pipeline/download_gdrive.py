#!/usr/bin/env python3
"""Download BCI dataset from Google Drive using OAuth.

Authenticates via browser, then downloads Phase 1 and Phase 2 folders.
Replaces sub03 and sub05 in Phase 1 with their Phase 2 versions.

Usage:
    python download_gdrive.py

First run will open a browser for Google login. Token is cached for
subsequent runs.
"""

import io
import os
import sys
import shutil
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PIPELINE_DIR = Path(__file__).parent
CREDENTIALS_FILE = list(PIPELINE_DIR.glob("client_secret_*.json"))[0]
TOKEN_FILE = PIPELINE_DIR / "token.json"

ROOT_FOLDER_ID = "1bHQ3Cm5NYfPvOb-EGZXeE6YcODsiLqth"

DATA_DIR = PIPELINE_DIR.parent / "bci" / "data"

# sub03 and sub05 should come from Phase 2 instead of Phase 1
PHASE2_REPLACEMENTS = {"sub03", "sub05"}


def authenticate():
    """Authenticate with Google and return Drive service."""
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def list_folder(service, folder_id):
    """List all files and subfolders in a Drive folder.

    Returns list of {id, name, mimeType}.
    """
    items = []
    page_token = None

    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return items


def download_file(service, file_id, dest_path):
    """Download a single file from Drive."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def download_folder_recursive(service, folder_id, dest_dir, indent=0):
    """Recursively download a Drive folder."""
    items = list_folder(service, folder_id)
    prefix = "  " * indent

    for item in sorted(items, key=lambda x: x["name"]):
        name = item["name"]
        if item["mimeType"] == "application/vnd.google-apps.folder":
            print(f"{prefix}[folder] {name}/")
            sub_dir = dest_dir / name
            download_folder_recursive(service, item["id"], sub_dir, indent + 1)
        else:
            dest_path = dest_dir / name
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"{prefix}[skip]   {name} (already exists)")
            else:
                print(f"{prefix}[dl]     {name}", end="", flush=True)
                download_file(service, item["id"], dest_path)
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                print(f" ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("BCI Dataset — Google Drive Downloader")
    print("=" * 60)

    print("\nAuthenticating...")
    service = authenticate()
    print("Authenticated.\n")

    # List root folder to find Phase 1 and Phase 2
    root_items = list_folder(service, ROOT_FOLDER_ID)
    print("Root folder contents:")
    for item in root_items:
        kind = "folder" if "folder" in item["mimeType"] else "file"
        print(f"  [{kind}] {item['name']}")

    phase1_folder = None
    phase2_folder = None
    for item in root_items:
        name_lower = item["name"].lower().replace(" ", "").replace("_", "")
        if "phase1" in name_lower or "phase 1" in item["name"].lower():
            phase1_folder = item
        elif "phase2" in name_lower or "phase 2" in item["name"].lower():
            phase2_folder = item

    if not phase1_folder:
        print("\nWARNING: Could not identify 'Phase 1' folder.")
        print("Available folders:", [i["name"] for i in root_items if "folder" in i["mimeType"]])
        # Try downloading everything into a temp dir for inspection
        temp_dir = PIPELINE_DIR / "_gdrive_raw"
        print(f"\nDownloading everything to {temp_dir} for inspection...")
        download_folder_recursive(service, ROOT_FOLDER_ID, temp_dir)
        print(f"\nDone. Check {temp_dir} and adjust the script.")
        return

    # Step 1: Download Phase 1 to data dir
    print(f"\n{'=' * 60}")
    print(f"Downloading Phase 1: {phase1_folder['name']}")
    print("=" * 60)

    phase1_items = list_folder(service, phase1_folder["id"])

    for item in sorted(phase1_items, key=lambda x: x["name"]):
        if item["mimeType"] != "application/vnd.google-apps.folder":
            continue

        sub_name = item["name"]

        # Skip sub03 and sub05 from Phase 1 — we'll get them from Phase 2
        if sub_name in PHASE2_REPLACEMENTS:
            print(f"\n  [skip Phase 1] {sub_name} — will use Phase 2 version")
            continue

        sub_dir = DATA_DIR / sub_name
        print(f"\n  Downloading {sub_name} → {sub_dir}")
        download_folder_recursive(service, item["id"], sub_dir, indent=2)

    # Step 2: Download Phase 2 replacements (sub03, sub05)
    if phase2_folder:
        print(f"\n{'=' * 60}")
        print(f"Downloading Phase 2 replacements: {phase2_folder['name']}")
        print("=" * 60)

        phase2_items = list_folder(service, phase2_folder["id"])

        for item in sorted(phase2_items, key=lambda x: x["name"]):
            if item["mimeType"] != "application/vnd.google-apps.folder":
                continue

            sub_name = item["name"]
            if sub_name not in PHASE2_REPLACEMENTS:
                continue

            sub_dir = DATA_DIR / sub_name

            # Remove old Phase 1 data if it exists
            if sub_dir.exists():
                print(f"\n  Removing old Phase 1 {sub_name}...")
                shutil.rmtree(sub_dir)

            print(f"  Downloading Phase 2 {sub_name} → {sub_dir}")
            download_folder_recursive(service, item["id"], sub_dir, indent=2)
    else:
        print("\nWARNING: Phase 2 folder not found — sub03 and sub05 NOT replaced")

    # Summary
    print(f"\n{'=' * 60}")
    print("Download complete")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    for sub_dir in sorted(DATA_DIR.glob("sub*")):
        eeg_files = list(sub_dir.glob("eeg_data_*.csv"))
        trial_files = list(sub_dir.glob("trial_log_*.csv"))
        source = "Phase 2" if sub_dir.name in PHASE2_REPLACEMENTS else "Phase 1"
        print(f"  {sub_dir.name} ({source}): "
              f"{len(eeg_files)} EEG files, {len(trial_files)} trial logs")

    print(f"\nNext step: cd ml_pipeline && python train.py")


if __name__ == "__main__":
    main()
