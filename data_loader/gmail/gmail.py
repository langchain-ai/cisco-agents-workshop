"""
Gmail API Integration Module for Email Assistant

This module provides functions and tools for interacting with Gmail and Google Calendar APIs.
It enables fetching emails, sending replies, scheduling calendar events, and checking calendar
availability. The module is designed to work with the email assistant to process real emails
from Gmail.

Features:
- OAuth2 authentication with Gmail and Calendar APIs
- Email fetching with configurable time range
- Email parsing and content extraction
- Email sending with thread support
- Calendar event retrieval and formatting
- Calendar invitation creation with Google Meet integration

Requirements:
- Google API credentials in .secrets/secrets.json
- OAuth token will be stored in .secrets/token.json
- Required scopes: Gmail modify and Calendar access
"""

import logging
import os
import sys
import base64
import email.utils
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Union, Set

# Third-party imports
import pytz
from dateutil import parser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# LangChain imports
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.email_assistant.schemas import EmailData

# Configure logging
logger = logging.getLogger(__name__)

# Google API configuration
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",  # Permission to read and modify Gmail
    "https://www.googleapis.com/auth/calendar",      # Permission to read and write Calendar
]

# Path configuration for authentication files
_ROOT = Path(__file__).parent.absolute()
_PORT = 54191  # Port for OAuth flow
_SECRETS_DIR = _ROOT / ".secrets"
_SECRETS_PATH = str(_SECRETS_DIR / "secrets.json")
_TOKEN_PATH = str(_SECRETS_DIR / "token.json")


def get_credentials(
    gmail_token: Optional[str] = None, gmail_secret: Optional[str] = None
) -> Credentials:
    """
    Get or create Google API credentials for Gmail and Calendar access.
    
    This function handles the OAuth2 authentication flow for Google APIs. It will:
    1. Create the secrets directory if it doesn't exist
    2. Check for token/secret from parameters or environment variables
    3. Load existing credentials if possible
    4. Refresh expired credentials if possible
    5. Initiate OAuth flow in browser if necessary
    
    Args:
        gmail_token: Optional token string, can also be set via GMAIL_TOKEN env var
        gmail_secret: Optional client secrets string, can also be set via GMAIL_SECRET env var
        
    Returns:
        Google OAuth2 Credentials object that can be used with API clients
        
    Note:
        If credentials don't exist or are invalid, this will launch a browser window
        for the user to authenticate with Google.
    """
    creds = None
    
    # Create secrets directory if it doesn't exist
    _SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for token/secret from parameters or environment variables
    gmail_token = gmail_token or os.getenv("GMAIL_TOKEN")
    if gmail_token:
        with open(_TOKEN_PATH, "w") as token:
            token.write(gmail_token)
            
    gmail_secret = gmail_secret or os.getenv("GMAIL_SECRET")
    if gmail_secret:
        with open(_SECRETS_PATH, "w") as secret:
            secret.write(gmail_secret)
            
    # Try to load existing credentials
    if os.path.exists(_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(_TOKEN_PATH)

    # Handle invalid, expired, or missing credentials
    if not creds or not creds.valid or not creds.has_scopes(_SCOPES):
        # Try to refresh if possible
        if (
            creds
            and creds.expired
            and creds.refresh_token
            and creds.has_scopes(_SCOPES)
        ):
            creds.refresh(Request())
        # Otherwise initiate OAuth flow
        else:
            flow = InstalledAppFlow.from_client_secrets_file(_SECRETS_PATH, _SCOPES)
            creds = flow.run_local_server(port=_PORT)
            
        # Save the refreshed/new credentials
        with open(_TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return creds


def extract_message_part(msg: Dict[str, Any]) -> str:
    """
    Recursively walk through the email parts to find message body.
    
    Gmail messages can have multiple parts with different MIME types.
    This function tries to find and decode the message content, preferring
    plain text over HTML when available.
    
    Args:
        msg: A message part dictionary from the Gmail API
        
    Returns:
        Decoded message body as string, or placeholder if no content found
    """
    # Try to extract plain text content
    if msg["mimeType"] == "text/plain":
        body_data = msg.get("body", {}).get("data")
        if body_data:
            return base64.urlsafe_b64decode(body_data).decode("utf-8")
            
    # Fall back to HTML content if plain text not available
    elif msg["mimeType"] == "text/html":
        body_data = msg.get("body", {}).get("data")
        if body_data:
            return base64.urlsafe_b64decode(body_data).decode("utf-8")
            
    # If message has multiple parts, recursively check each part
    if "parts" in msg:
        for part in msg["parts"]:
            body = extract_message_part(part)
            if body:
                return body
                
    # Return placeholder if no content found
    return "No message body available."


def parse_time(send_time: str) -> datetime:
    """
    Parse an email timestamp string into a datetime object.
    
    Args:
        send_time: Time string in standard email format (RFC 2822)
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If the time string cannot be parsed
    """
    try:
        parsed_time = parser.parse(send_time)
        return parsed_time
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parsing time: {send_time} - {e}")


def create_message(
    sender: str, 
    to: List[str], 
    subject: str, 
    message_text: str, 
    thread_id: str, 
    original_message_id: str
) -> Dict[str, Any]:
    """
    Create a Gmail API-compatible message object for sending emails.
    
    This function creates an email message as a reply to an existing thread,
    with proper headers for threading.
    
    Args:
        sender: Email address of sender (usually 'me')
        to: List of recipient email addresses
        subject: Email subject line
        message_text: Email body content
        thread_id: Gmail thread ID to associate this message with
        original_message_id: Message-ID of the original email being replied to
        
    Returns:
        Dictionary formatted for Gmail API message.send method
    """
    # Create multipart MIME message
    message = MIMEMultipart()
    message["to"] = ", ".join(to)
    message["from"] = sender
    message["subject"] = subject
    
    # Add threading headers
    message["In-Reply-To"] = original_message_id
    message["References"] = original_message_id
    message["Message-ID"] = email.utils.make_msgid()
    
    # Add message content
    msg = MIMEText(message_text)
    message.attach(msg)
    
    # Convert to Gmail API format
    raw = base64.urlsafe_b64encode(message.as_bytes())
    raw = raw.decode()
    
    return {"raw": raw, "threadId": thread_id}


def get_recipients(
    headers: List[Dict[str, str]],
    email_address: str,
    addn_receipients: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract recipients from email headers and add additional recipients.
    
    This function extracts email addresses from the To and CC fields of an email,
    adds the original sender, and removes the current user's email address.
    
    Args:
        headers: List of email header dictionaries from Gmail API
        email_address: Current user's email address (to exclude from recipients)
        addn_receipients: Optional additional recipients to include
        
    Returns:
        List of email addresses to send a reply to
    """
    # Initialize recipient set with additional recipients if provided
    recipients: Set[str] = set(addn_receipients or [])
    sender = None
    
    # Extract addresses from headers
    for header in headers:
        # Add all To: and CC: addresses
        if header["name"].lower() in ["to", "cc"]:
            recipients.update(header["value"].replace(" ", "").split(","))
        # Save the sender address
        if header["name"].lower() == "from":
            sender = header["value"]
            
    # Ensure the original sender is included in the response
    if sender:
        recipients.add(sender)
        
    # Remove the current user's email address to avoid sending to self
    for r in list(recipients):
        if email_address in r:
            recipients.remove(r)
            
    return list(recipients)


def send_message(service: Any, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message via the Gmail API.
    
    Args:
        service: Gmail API service object
        user_id: User ID to send as (usually 'me')
        message: Message object formatted for Gmail API
        
    Returns:
        Response from Gmail API containing sent message details
    """
    message = service.users().messages().send(userId=user_id, body=message).execute()
    return message


def send_email(
    email_id: str,
    response_text: str,
    email_address: str,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
    addn_receipients: Optional[List[str]] = None,
) -> None:
    """
    Send a reply to an existing email thread.
    
    This function fetches the original email by ID, determines the recipients,
    creates a reply message, and sends it through the Gmail API.
    
    Args:
        email_id: Gmail message ID to reply to
        response_text: Content of the reply
        email_address: Current user's email address
        gmail_token: Optional token for Gmail API authentication
        gmail_secret: Optional credentials for Gmail API authentication
        addn_receipients: Optional additional recipients to include
    """
    creds = get_credentials(gmail_token, gmail_secret)

    service = build("gmail", "v1", credentials=creds)
    message = service.users().messages().get(userId="me", id=email_id).execute()

    headers = message["payload"]["headers"]
    message_id = next(
        header["value"] for header in headers if header["name"].lower() == "message-id"
    )
    thread_id = message["threadId"]

    # Get recipients and sender
    recipients = get_recipients(headers, email_address, addn_receipients)

    # Create the response
    subject = next(
        header["value"] for header in headers if header["name"].lower() == "subject"
    )
    response_subject = subject
    response_message = create_message(
        "me", recipients, response_subject, response_text, thread_id, message_id
    )
    # Send the response
    send_message(service, "me", response_message)


def fetch_group_emails(
    to_email: str,
    minutes_since: int = 30,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
) -> Iterable[EmailData]:
    """
    Fetch recent emails from Gmail that involve the specified email address.
    
    This function retrieves emails where the specified address is either a sender
    or recipient, processes them, and returns them in a format suitable for the
    email assistant to process.
    
    Args:
        to_email: Email address to fetch messages for
        minutes_since: Only retrieve emails newer than this many minutes
        gmail_token: Optional token for Gmail API authentication
        gmail_secret: Optional credentials for Gmail API authentication
        
    Yields:
        EmailData objects containing processed email information
    """
    # Get Gmail API credentials
    creds = get_credentials(gmail_token, gmail_secret)
    service = build("gmail", "v1", credentials=creds)
    
    # Calculate timestamp for filtering
    after = int((datetime.now() - timedelta(minutes=minutes_since)).timestamp())
    
    # Construct Gmail search query
    query = f"(to:{to_email} OR from:{to_email}) after:{after}"
    
    # Retrieve all matching messages (handling pagination)
    messages = []
    nextPageToken = None
    logger.info(f"Fetching emails for {to_email} from last {minutes_since} minutes")
    
    while True:
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, pageToken=nextPageToken)
            .execute()
        )
        if "messages" in results:
            messages.extend(results["messages"])
        nextPageToken = results.get("nextPageToken")
        if not nextPageToken:
            break

    # Process each message
    count = 0
    for message in messages:
        try:
            # Get full message details
            msg = service.users().messages().get(userId="me", id=message["id"]).execute()
            thread_id = msg["threadId"]
            payload = msg["payload"]
            headers = payload.get("headers", [])
            
            # Get thread details to determine conversation context
            thread = service.users().threads().get(userId="me", id=thread_id).execute()
            messages_in_thread = thread["messages"]
            
            # Analyze the last message in the thread to determine if we need to process it
            last_message = messages_in_thread[-1]
            last_headers = last_message["payload"]["headers"]
            
            # Get sender of last message
            from_header = next(
                header["value"] for header in last_headers if header["name"] == "From"
            )
            last_from_header = next(
                header["value"]
                for header in last_message["payload"].get("headers")
                if header["name"] == "From"
            )
            
            # If the last message was sent by the user, mark this as a user response
            # and don't process it further (assistant doesn't need to respond to user's own emails)
            if to_email in last_from_header:
                yield {
                    "id": message["id"],
                    "thread_id": message["threadId"],
                    "user_respond": True,
                }
                continue
                
            # Only process messages that are the latest in their thread and not from the user
            if to_email not in from_header and message["id"] == last_message["id"]:
                # Extract email metadata from headers
                subject = next(
                    header["value"] for header in headers if header["name"] == "Subject"
                )
                from_email = next(
                    (header["value"] for header in headers if header["name"] == "From"),
                    "",
                ).strip()
                _to_email = next(
                    (header["value"] for header in headers if header["name"] == "To"),
                    "",
                ).strip()
                
                # Use Reply-To header if present
                if reply_to := next(
                    (
                        header["value"]
                        for header in headers
                        if header["name"] == "Reply-To"
                    ),
                    "",
                ).strip():
                    from_email = reply_to
                    
                # Extract and parse email timestamp
                send_time = next(
                    header["value"] for header in headers if header["name"] == "Date"
                )
                parsed_time = parse_time(send_time)
                
                # Extract email body content
                body = extract_message_part(payload)
                
                # Yield the processed email data
                yield {
                    "from_email": from_email,
                    "to_email": _to_email,
                    "subject": subject,
                    "page_content": body,
                    "id": message["id"],
                    "thread_id": message["threadId"],
                    "send_time": parsed_time.isoformat(),
                }
                count += 1
                
        except Exception as e:
            logger.warning(f"Failed to process message {message['id']}: {str(e)}")

    logger.info(f"Found {count} emails to process.")


def mark_as_read(
    message_id: str,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
) -> None:
    """
    Mark a Gmail message as read by removing the UNREAD label.
    
    Args:
        message_id: Gmail message ID to mark as read
        gmail_token: Optional token for Gmail API authentication
        gmail_secret: Optional credentials for Gmail API authentication
    """
    creds = get_credentials(gmail_token, gmail_secret)
    service = build("gmail", "v1", credentials=creds)
    
    service.users().messages().modify(
        userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()
    
    logger.info(f"Marked message {message_id} as read")


class CalInput(BaseModel):
    """
    Input schema for the get_events_for_days tool.
    
    Attributes:
        date_strs: List of days for which to retrieve events in dd-mm-yyyy format
    """
    date_strs: List[str] = Field(
        description="The days for which to retrieve events. Each day should be represented by dd-mm-yyyy string."
    )


@tool(args_schema=CalInput)
def get_events_for_days(date_strs: List[str]) -> str:
    """
    Retrieves calendar events for a list of days.
    
    This tool fetches events from Google Calendar for specified days and formats them
    for display.
    
    Input format: ['dd-mm-yyyy', 'dd-mm-yyyy']
    
    Args:
        date_strs: The days for which to retrieve events (dd-mm-yyyy format)
        
    Returns:
        Formatted string of events for each requested day
    """
    # Get Google Calendar API credentials
    creds = get_credentials(None, None)
    service = build("calendar", "v3", credentials=creds)
    
    results = ""
    for date_str in date_strs:
        logger.info(f"Fetching calendar events for {date_str}")
        
        # Convert the date string to a datetime.date object
        day = datetime.strptime(date_str, "%d-%m-%Y").date()

        # Set time range for the entire day
        start_of_day = datetime.combine(day, time.min).isoformat() + "Z"
        end_of_day = datetime.combine(day, time.max).isoformat() + "Z"

        # Fetch events for this day
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_of_day,
                timeMax=end_of_day,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        # Format the events and add them to the results
        results += f"***FOR DAY {date_str}***\n\n" + print_events(events)
        
    return results


def format_datetime_with_timezone(dt_str: str, timezone: str = "US/Pacific") -> str:
    """
    Format a datetime string with the specified timezone.
    
    Converts an ISO format datetime string to a human-readable format with
    the specified timezone.
    
    Args:
        dt_str: The datetime string to format (ISO format)
        timezone: The timezone to use for formatting (default: US/Pacific)
        
    Returns:
        A formatted datetime string with the timezone abbreviation
    """
    # Convert ISO format string to datetime, handling Z suffix
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    
    # Convert to specified timezone
    tz = pytz.timezone(timezone)
    dt = dt.astimezone(tz)
    
    # Format as human-readable string
    return dt.strftime("%Y-%m-%d %I:%M %p %Z")


def print_events(events: List[Dict[str, Any]]) -> str:
    """
    Format a list of calendar events into a human-readable string.
    
    Args:
        events: List of Google Calendar event objects
        
    Returns:
        Formatted string representation of events
    """
    if not events:
        return "No events found for this day."

    result = ""

    for event in events:
        # Get event details, falling back to all-day format if no time specified
        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event["end"].get("dateTime", event["end"].get("date"))
        summary = event.get("summary", "No Title")

        # Format datetime strings (if not all-day events)
        if "T" in start:  # "T" indicates a datetime (not just a date)
            start = format_datetime_with_timezone(start)
            end = format_datetime_with_timezone(end)

        # Format the event information
        result += f"Event: {summary}\n"
        result += f"Starts: {start}\n"
        result += f"Ends: {end}\n"
        result += "-" * 40 + "\n"
        
    return result


def send_calendar_invite(
    emails: List[str], 
    title: str, 
    start_time: str, 
    end_time: str, 
    email_address: str, 
    timezone: str = "PST"
) -> bool:
    """
    Create and send a Google Calendar invitation.
    
    This function creates a calendar event with Google Meet integration
    and sends invitations to the specified attendees.
    
    Args:
        emails: List of email addresses to invite
        title: Event title/summary
        start_time: Event start time (ISO format)
        end_time: Event end time (ISO format)
        email_address: Email address of the event organizer
        timezone: Timezone for the event (default: PST)
        
    Returns:
        Boolean indicating success or failure
    """
    # Get Google Calendar API credentials
    creds = get_credentials(None, None)
    service = build("calendar", "v3", credentials=creds)

    # Parse the start and end times
    start_datetime = datetime.fromisoformat(start_time)
    end_datetime = datetime.fromisoformat(end_time)
    
    # Ensure the organizer is included in the attendee list
    emails = list(set(emails + [email_address]))
    
    # Create event object with Google Meet integration
    event = {
        "summary": title,
        "start": {
            "dateTime": start_datetime.isoformat(),
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_datetime.isoformat(),
            "timeZone": timezone,
        },
        "attendees": [{"email": email} for email in emails],
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 24 * 60},  # Email reminder 1 day before
                {"method": "popup", "minutes": 10},       # Popup reminder 10 minutes before
            ],
        },
        # Create a Google Meet conference link
        "conferenceData": {
            "createRequest": {
                "requestId": f"{title}-{start_datetime.isoformat()}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }

    try:
        # Insert the event and send notifications to attendees
        calendar_event = service.events().insert(
            calendarId="primary",
            body=event,
            sendNotifications=True,
            conferenceDataVersion=1,  # Enable conference creation
        ).execute()
        
        logger.info(f"Created calendar event: {title}")
        return True
    except Exception as e:
        logger.error(f"Failed to create calendar event: {e}")
        return False
