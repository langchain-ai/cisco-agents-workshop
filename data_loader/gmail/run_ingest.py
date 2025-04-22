"""
Gmail Ingestion Script for LangGraph Email Assistant

This script fetches emails from Gmail and processes them using a LangGraph
email assistant. It connects to the Gmail API, retrieves recent emails,
and passes them to the specified LangGraph for processing. Processed emails
are also logged to a JSON file for record-keeping.

Usage:
    python run_ingest.py [options]

Options include specifying the email address, time range, LangGraph URL,
graph name, and more. See --help for details.
"""

import sys
import os
# Add project root to path to allow imports from project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from data_loader.gmail.gmail import fetch_group_emails
from langgraph_sdk import get_client
import httpx
import uuid
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main(
    url: Optional[str] = None,
    minutes_since: int = 60,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
    early: bool = True,
    rerun: bool = False,
    email: Optional[str] = None,
    log_dir: str = "email_logs",
    graph_name: str = "email_assistant_hitl_memory",
) -> None:
    """
    Main function to fetch emails from Gmail and process them with a LangGraph.
    
    Args:
        url: URL of the LangGraph deployment (default: http://127.0.0.1:2024)
        minutes_since: Only process emails newer than this many minutes (default: 60)
        gmail_token: Optional token for Gmail API authentication
        gmail_secret: Optional credentials for Gmail API authentication
        early: Whether to exit when encountering an already processed email (default: True)
        rerun: Whether to reprocess emails even if they've been seen before (default: False)
        email: Email address to fetch messages from
        log_dir: Directory to store email logs (default: "email_logs")
        graph_name: Name of the LangGraph to use (default: "email_assistant_hitl_memory")
    
    Returns:
        None
    """
    # Use the provided email or get from environment variable
    if email is None:
        email_address = os.environ.get("EMAIL_ADDRESS")
        if email_address is None:
            raise ValueError("No email address provided. Set EMAIL_ADDRESS environment variable or use --email argument.")
    else:
        email_address = email
        
    # Initialize LangGraph client with appropriate URL
    if url is None:
        # Connect to local LangGraph server
        client = get_client(url="http://127.0.0.1:2024")
    else:
        client = get_client(url=url)
    
    # Set the graph to the provided graph_name
    client.graph = graph_name
    logger.info(f"Connected to graph: {graph_name}")
    
    # Set up logging directory and file
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"email_dump_{timestamp}.json")
    logger.info(f"Will dump emails to {log_file}")
    
    # Initialize list to store emails for logging
    emails_to_log = []

    # Fetch and process emails from Gmail
    # TODO: This really should be async
    for email in fetch_group_emails(
        email_address,
        minutes_since=minutes_since,
        gmail_token=gmail_token,
        gmail_secret=gmail_secret,
    ):
        # Add email to log list for record-keeping
        emails_to_log.append(email)
        
        # Create a deterministic thread ID from the email thread_id
        thread_id = str(
            uuid.UUID(hex=hashlib.md5(email["thread_id"].encode("UTF-8")).hexdigest())
        )
        
        # Check if thread exists in LangGraph
        try:
            thread_info = await client.threads.get(thread_id)
        except httpx.HTTPStatusError as e:
            # Skip if this is a user response and thread doesn't exist
            if "user_respond" in email:
                continue
                
            # Create new thread if it doesn't exist
            if e.response.status_code == 404:
                thread_info = await client.threads.create(thread_id=thread_id)
            else:
                # Re-raise unexpected errors
                raise e
                
        # Handle user responses separately
        if "user_respond" in email:
            # Mark the thread as complete
            await client.threads.update_state(thread_id, None, as_node="__end__")
            continue
            
        # Check if we've already processed this email
        recent_email = thread_info["metadata"].get("email_id")
        if recent_email == email["id"]:
            if early:
                # Exit early if we've hit a previously processed email
                break
            else:
                if not rerun:
                    # Skip already processed emails unless rerun is enabled
                    continue
                    
        # Update thread metadata with current email ID to track progress
        await client.threads.update(thread_id, metadata={"email_id": email["id"]})

        # Process the email with the specified LangGraph
        await client.runs.create(
            thread_id,
            graph_name,
            input={"email_input": email},
            multitask_strategy="rollback",
        )
    
    # Write all processed emails to log file for record-keeping
    if emails_to_log:
        logger.info(f"Writing {len(emails_to_log)} emails to {log_file}")
        with open(log_file, "w") as f:
            json.dump(emails_to_log, f, indent=2, default=str)
        logger.info(f"Email dump completed successfully to {log_file}")
    else:
        logger.info("No emails collected during this run")


if __name__ == "__main__":
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Fetch emails from Gmail and process them with a LangGraph email assistant"
    )
    
    # LangGraph connection parameters
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of the LangGraph deployment (default: http://127.0.0.1:2024)",
    )
    parser.add_argument(
        "--graph-name",
        type=str,
        default="email_assistant_hitl_memory",
        help="Name of the LangGraph to use",
    )
    
    # Email processing parameters
    parser.add_argument(
        "--early",
        type=int,
        default=1,
        help="Whether to stop when encountering previously processed emails (1=yes, 0=no)",
    )
    parser.add_argument(
        "--rerun",
        type=int,
        default=0,
        help="Whether to reprocess emails even if already seen (1=yes, 0=no)",
    )
    parser.add_argument(
        "--minutes-since",
        type=int,
        default=60,
        help="Only process emails that are less than this many minutes old",
    )
    
    # Gmail API authentication parameters
    parser.add_argument(
        "--gmail-token",
        type=str,
        default=None,
        help="Token to use for Gmail API authentication",
    )
    parser.add_argument(
        "--gmail-secret",
        type=str,
        default=None,
        help="Client secrets for Gmail API authentication",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email address to fetch messages from (alternative to EMAIL_ADDRESS env var)",
    )
    
    # Logging parameters
    parser.add_argument(
        "--log-dir",
        type=str,
        default="email_logs",
        help="Directory to store email logs",
    )

    # Parse arguments and run the main function
    args = parser.parse_args()
    asyncio.run(
        main(
            url=args.url,
            minutes_since=args.minutes_since,
            gmail_token=args.gmail_token,
            gmail_secret=args.gmail_secret,
            early=bool(args.early),
            rerun=bool(args.rerun),
            email=args.email,
            log_dir=args.log_dir,
            graph_name=args.graph_name,
        )
    )
