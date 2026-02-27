"""
Authentication Module for Auto DataLab
Handles user authentication, session management, and user credentials.
"""

import hashlib
import json
import os
import base64
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import streamlit as st


class AuthManager:
    """Manages user authentication and session handling."""
    
    def __init__(self, users_file: str = "users.json"):
        """
        Initialize the authentication manager.
        
        Args:
            users_file: Path to the JSON file storing user credentials
        """
        self.users_file = Path(users_file)
        self._ensure_users_file()
        
    def _ensure_users_file(self):
        """Create users file if it doesn't exist with default admin user."""
        if not self.users_file.exists():
            default_users = {
                "admin": {
                    "password": self._hash_password("admin123"),
                    "role": "admin",
                    "email": "admin@autodatalab.com",
                    "created_at": datetime.now().isoformat(),
                    "full_name": "Administrator",
                    "bio": "System Administrator",
                    "phone": "",
                    "address": "",
                    "education": "",
                    "profile_picture_data": None,
                    "department": "",
                    "job_title": ""
                }
            }
            self._save_users(default_users)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self) -> Dict:
        """Load users from JSON file."""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading users: {e}")
            return {}
    
    def _save_users(self, users: Dict):
        """Save users to JSON file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=4)
        except Exception as e:
            st.error(f"Error saving users: {e}")
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Tuple of (success, user_info)
        """
        users = self._load_users()
        
        if username not in users:
            return False, None
        
        user = users[username]
        if user['password'] == self._hash_password(password):
            return True, {
                'username': username,
                'role': user.get('role', 'user'),
                'email': user.get('email', ''),
                'full_name': user.get('full_name', username),
                'bio': user.get('bio', ''),
                'phone': user.get('phone', ''),
                'address': user.get('address', ''),
                'education': user.get('education', ''),
                'profile_picture_data': user.get('profile_picture_data', None),
                'department': user.get('department', ''),
                'job_title': user.get('job_title', '')
            }
        
        return False, None
    
    def register_user(self, username: str, password: str, email: str, 
                     full_name: str, role: str = "user") -> Tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Username
            password: Plain text password
            email: User email
            full_name: User's full name
            role: User role (user/admin)
            
        Returns:
            Tuple of (success, message)
        """
        users = self._load_users()
        
        if username in users:
            return False, "Username already exists"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        users[username] = {
            'password': self._hash_password(password),
            'role': role,
            'email': email,
            'full_name': full_name,
            'created_at': datetime.now().isoformat(),
            'bio': '',
            'phone': '',
            'address': '',
            'education': '',
            'profile_picture_data': None,
            'department': '',
            'job_title': ''
        }
        
        self._save_users(users)
        return True, "User registered successfully"
    
    def change_password(self, username: str, old_password: str, 
                       new_password: str) -> Tuple[bool, str]:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        users = self._load_users()
        
        if username not in users:
            return False, "User not found"
        
        if users[username]['password'] != self._hash_password(old_password):
            return False, "Incorrect current password"
        
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"
        
        users[username]['password'] = self._hash_password(new_password)
        self._save_users(users)
        
        return True, "Password changed successfully"
    
    def delete_user(self, username: str) -> Tuple[bool, str]:
        """
        Delete a user.
        
        Args:
            username: Username to delete
            
        Returns:
            Tuple of (success, message)
        """
        if username == "admin":
            return False, "Cannot delete admin user"
        
        users = self._load_users()
        
        if username not in users:
            return False, "User not found"
        
        del users[username]
        self._save_users(users)
        
        return True, "User deleted successfully"
    
    def update_profile(self, username: str, profile_data: Dict) -> Tuple[bool, str]:
        """
        Update user profile information.
        
        Args:
            username: Username
            profile_data: Dictionary with profile fields to update
            
        Returns:
            Tuple of (success, message)
        """
        users = self._load_users()
        
        if username not in users:
            return False, "User not found"
        
        # Allowed fields to update (cannot update password, username, or role here)
        allowed_fields = ['full_name', 'email', 'bio', 'phone', 'address', 
                         'education', 'profile_picture_data', 'department', 'job_title']
        
        for field, value in profile_data.items():
            if field in allowed_fields:
                users[username][field] = value
        
        self._save_users(users)
        return True, "Profile updated successfully"
    
    def get_user_profile(self, username: str) -> Optional[Dict]:
        """
        Get complete user profile.
        
        Args:
            username: Username
            
        Returns:
            User profile dictionary (without password) or None
        """
        users = self._load_users()
        
        if username not in users:
            return None
        
        user_data = users[username].copy()
        user_data.pop('password', None)  # Remove password
        user_data['username'] = username
        
        return user_data
    def get_all_users(self) -> Dict:
        """Get all users (excluding passwords)."""
        users = self._load_users()
        safe_users = {}
        
        for username, data in users.items():
            safe_users[username] = {
                'role': data.get('role', 'user'),
                'email': data.get('email', ''),
                'full_name': data.get('full_name', username),
                'created_at': data.get('created_at', '')
            }
        
        return safe_users


def init_session_state():
    """Initialize session state variables for authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)


def get_current_user() -> Optional[Dict]:
    """Get current logged in user info."""
    return st.session_state.get('user_info', None)


def logout():
    """Logout current user."""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.login_time = None


def login_page():
    """Display login page."""
    st.markdown("""
        <style>
        .login-container {
            max-width: 450px;
            margin: 0 auto;
            padding: 3rem 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin-top: 5rem;
        }
        .login-title {
            text-align: center;
            color: #1F3A8A;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .login-subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Tablet Media Query (768px - 1023px) */
        @media screen and (min-width: 768px) and (max-width: 1023px) {
            .login-container {
                max-width: 400px;
                padding: 2.5rem 1.5rem;
                margin-top: 3rem;
            }
            .login-title {
                font-size: 1.8rem;
                margin-bottom: 0.35rem;
            }
            .login-subtitle {
                font-size: 0.9rem;
                margin-bottom: 1.5rem;
            }
        }
        
        /* Mobile Media Query (600px - 767px) */
        @media screen and (min-width: 600px) and (max-width: 767px) {
            .login-container {
                max-width: 95%;
                padding: 2rem 1.25rem;
                margin-top: 2rem;
                border-radius: 8px;
            }
            .login-title {
                font-size: 1.6rem;
                margin-bottom: 0.3rem;
            }
            .login-subtitle {
                font-size: 0.85rem;
                margin-bottom: 1.25rem;
            }
        }
        
        /* Small Mobile Media Query (below 600px) */
        @media screen and (max-width: 599px) {
            .login-container {
                max-width: 100%;
                padding: 1.5rem 1rem;
                margin-top: 1rem;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }
            .login-title {
                font-size: 1.4rem;
                margin-bottom: 0.25rem;
            }
            .login-subtitle {
                font-size: 0.8rem;
                margin-bottom: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown('<h1 class="login-title">[AUTODATALAB]</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">End-to-End Data Science Platform</p>', unsafe_allow_html=True)
        
        # Create tabs for login and register
        tab1, tab2 = st.tabs(["[LOGIN]", "[REGISTER]"])
        
        auth_manager = AuthManager()
        
        with tab1:
            st.markdown("### Welcome Back")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        success, user_info = auth_manager.authenticate(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user_info = user_info
                            st.session_state.login_time = datetime.now()
                            st.success(f"Welcome back, {user_info['full_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
            
            st.info("Default credentials: **admin** / **admin123**")
        
        with tab2:
            st.markdown("### Create New Account")
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_full_name = st.text_input("Full Name", placeholder="Your full name")
                new_email = st.text_input("Email", placeholder="your.email@example.com")
                new_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
                new_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                register_submit = st.form_submit_button("Register", use_container_width=True)
                
                if register_submit:
                    if not all([new_username, new_full_name, new_email, new_password, new_password_confirm]):
                        st.error("Please fill in all fields")
                    elif new_password != new_password_confirm:
                        st.error("Passwords do not match")
                    else:
                        success, message = auth_manager.register_user(
                            new_username, new_password, new_email, new_full_name
                        )
                        if success:
                            st.success(message + " You can now login!")
                        else:
                            st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_user_profile():
    """Display user profile in sidebar."""
    user = get_current_user()
    if user:
        with st.sidebar:
            st.markdown("---")
            
            # Display profile picture if available
            if user.get('profile_picture_data'):
                try:
                    img_data = base64.b64decode(user['profile_picture_data'])
                    st.image(img_data, width=100)
                except Exception as e:
                    st.markdown(f"### 👤 {user['full_name']}")
            else:
                st.markdown(f"### 👤 {user['full_name']}")
            
            st.caption(f"**Role:** {user['role'].title()}")
            st.caption(f"**Email:** {user['email']}")
            
            if st.session_state.login_time:
                st.caption(f"**Logged in:** {st.session_state.login_time.strftime('%I:%M %p')}")
            
            if st.button("LOGOUT", use_container_width=True):
                logout()
                st.rerun()


def require_authentication(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper


def user_management_page():
    """Admin page for managing users."""
    user = get_current_user()
    
    if not user or user['role'] != 'admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    st.title("User Management")
    
    auth_manager = AuthManager()
    
    tab1, tab2, tab3 = st.tabs(["View Users", "Change Password", "User Profiles"])
    
    with tab1:
        st.subheader("Registered Users")
        users = auth_manager.get_all_users()
        
        if users:
            for username, info in users.items():
                with st.expander(f"{info['full_name']} (@{username})"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Role:** {info['role'].title()}")
                        st.write(f"**Email:** {info['email']}")
                        if info.get('department'):
                            st.write(f"**Department:** {info['department']}")
                        if info.get('job_title'):
                            st.write(f"**Job Title:** {info['job_title']}")
                        st.write(f"**Created:** {info.get('created_at', 'N/A')[:10]}")
                    with col2:
                        if username != "admin":
                            if st.button("Delete", key=f"del_{username}"):
                                success, msg = auth_manager.delete_user(username)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
        else:
            st.info("No users found")
    
    with tab2:
        st.subheader("Change Your Password")
        with st.form("change_password_form"):
            old_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password", type="password")
            new_pass_confirm = st.text_input("Confirm New Password", type="password")
            submit = st.form_submit_button("Change Password")
            
            if submit:
                if not all([old_pass, new_pass, new_pass_confirm]):
                    st.error("Please fill in all fields")
                elif new_pass != new_pass_confirm:
                    st.error("New passwords do not match")
                else:
                    success, msg = auth_manager.change_password(
                        user['username'], old_pass, new_pass
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    
    with tab3:
        st.subheader("Detailed User Profiles")
        
        users = auth_manager.get_all_users()
        
        if users:
            # Create a searchable list
            user_list = [f"{info['full_name']} (@{username})" 
                        for username, info in users.items()]
            selected = st.selectbox("Select User to View Profile", user_list)
            
            if selected:
                # Extract username from selection
                username = selected.split("(@")[1].rstrip(")")
                profile = auth_manager.get_user_profile(username)
                
                if profile:
                    st.markdown("---")
                    
                    # Profile header
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        # Display profile picture if available, otherwise show placeholder
                        if profile.get('profile_picture_data'):
                            try:
                                img_data = base64.b64decode(profile['profile_picture_data'])
                                st.image(img_data, width=150)
                            except Exception as e:
                                st.markdown(f"<div style='font-size: 3rem; text-align: center; color: #1F3A8A;'>[USER]</div>", 
                                          unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='font-size: 3rem; text-align: center; color: #1F3A8A;'>[USER]</div>", 
                                      unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"### {profile['full_name']}")
                        st.caption(f"@{profile['username']} • {profile['role'].upper()}")
                        if profile.get('job_title'):
                            st.write(f"**{profile['job_title']}**")
                        if profile.get('department'):
                            st.write(f"Department: {profile['department']}")
                    
                    st.markdown("---")
                    
                    # Contact Information
                    st.markdown("### Contact Information")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.write(f"**Email:** {profile.get('email', 'N/A')}")
                        st.write(f"**Phone:** {profile.get('phone', 'N/A') or 'Not provided'}")
                    with info_col2:
                        st.write(f"**Address:** {profile.get('address', 'Not provided') or 'Not provided'}")
                        st.write(f"**Joined:** {profile.get('created_at', 'N/A')[:10]}")
                    
                    st.markdown("---")
                    
                    # Bio
                    st.markdown("### Bio")
                    bio = profile.get('bio', '')
                    if bio:
                        st.info(bio)
                    else:
                        st.caption("No bio provided")
                    
                    # Education
                    st.markdown("### Education")
                    education = profile.get('education', '')
                    if education:
                        st.info(education)
                    else:
                        st.caption("No education information provided")
        else:
            st.info("No users found")


def edit_profile_page():
    """User profile editing page."""
    user = get_current_user()
    
    if not user:
        st.error("You must be logged in to edit your profile")
        return
    
    st.title("Edit Profile")
    st.caption(f"Update your profile information, @{user['username']}")
    
    auth_manager = AuthManager()
    current_profile = auth_manager.get_user_profile(user['username'])
    
    if not current_profile:
        st.error("Could not load profile")
        return
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Basic Info", "Contact & Bio", "Professional Info"])
    
    with tab1:
        st.subheader("Basic Information")
        
        with st.form("basic_info_form"):
            # Profile picture uploader
            st.markdown("**Upload Profile Picture**")
            st.info("Upload a profile picture (JPG, PNG, or GIF). Max size: 2MB")
            
            # Show current profile picture if available
            if current_profile.get('profile_picture_data'):
                st.markdown("**Current Picture:**")
                try:
                    img_data = base64.b64decode(current_profile['profile_picture_data'])
                    st.image(img_data, width=100)
                except Exception as e:
                    st.warning("Could not display current picture")
            else:
                st.caption("No picture uploaded yet")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a profile picture",
                type=["jpg", "jpeg", "png", "gif"],
                key="profile_pic_upload"
            )
            
            picture_data = None
            if uploaded_file is not None:
                # Check file size (2MB max)
                if uploaded_file.size > 2 * 1024 * 1024:
                    st.error("File size exceeds 2MB limit")
                else:
                    # Convert to base64
                    picture_data = base64.b64encode(uploaded_file.getvalue()).decode()
                    st.success("Picture selected successfully")
                    st.image(uploaded_file, width=100, caption="New picture preview")
            
            full_name = st.text_input("Full Name", value=current_profile.get('full_name', ''))
            email = st.text_input("Email", value=current_profile.get('email', ''))
            
            submit_basic = st.form_submit_button("Save Basic Info", use_container_width=True)
            
            if submit_basic:
                if not full_name or not email:
                    st.error("Full name and email are required")
                else:
                    update_data = {
                        'full_name': full_name,
                        'email': email
                    }
                    # Only update picture if a new one was uploaded
                    if picture_data is not None:
                        update_data['profile_picture_data'] = picture_data
                    
                    success, msg = auth_manager.update_profile(
                        user['username'],
                        update_data
                    )
                    if success:
                        st.success(msg)
                        # Update session state
                        st.session_state.user_info['full_name'] = full_name
                        st.session_state.user_info['email'] = email
                        if picture_data is not None:
                            st.session_state.user_info['profile_picture_data'] = picture_data
                        st.rerun()
                    else:
                        st.error(msg)
    
    with tab2:
        st.subheader("Contact & Bio")
        
        with st.form("contact_bio_form"):
            phone = st.text_input("Phone Number", value=current_profile.get('phone', ''),
                                 placeholder="+1 (555) 123-4567")
            address = st.text_area("Address", value=current_profile.get('address', ''),
                                  placeholder="123 Main St, City, State, ZIP")
            bio = st.text_area("Bio", value=current_profile.get('bio', ''),
                             placeholder="Tell us about yourself...",
                             height=150,
                             max_chars=500)
            st.caption(f"{len(bio)}/500 characters")
            
            submit_contact = st.form_submit_button("Save Contact & Bio", use_container_width=True)
            
            if submit_contact:
                success, msg = auth_manager.update_profile(
                    user['username'],
                    {
                        'phone': phone,
                        'address': address,
                        'bio': bio
                    }
                )
                if success:
                    st.success(msg)
                    st.session_state.user_info['phone'] = phone
                    st.session_state.user_info['address'] = address
                    st.session_state.user_info['bio'] = bio
                    st.rerun()
                else:
                    st.error(msg)
    
    with tab3:
        st.subheader("Professional Information")
        
        with st.form("professional_form"):
            job_title = st.text_input("Job Title", value=current_profile.get('job_title', ''),
                                     placeholder="e.g., Data Scientist")
            department = st.text_input("Department", value=current_profile.get('department', ''),
                                      placeholder="e.g., Analytics")
            education = st.text_area("Education", value=current_profile.get('education', ''),
                                   placeholder="List your degrees, certifications, etc.\ne.g.:\nBS Computer Science - MIT (2020)\nMS Data Science - Stanford (2022)",
                                   height=150)
            
            submit_professional = st.form_submit_button("Save Professional Info", use_container_width=True)
            
            if submit_professional:
                success, msg = auth_manager.update_profile(
                    user['username'],
                    {
                        'job_title': job_title,
                        'department': department,
                        'education': education
                    }
                )
                if success:
                    st.success(msg)
                    st.session_state.user_info['job_title'] = job_title
                    st.session_state.user_info['department'] = department
                    st.session_state.user_info['education'] = education
                    st.rerun()
                else:
                    st.error(msg)
    
    st.markdown("---")
    
    # Preview profile
    with st.expander("Preview Your Profile", expanded=False):
        # Show picture
        if current_profile.get('profile_picture_data'):
            try:
                img_data = base64.b64decode(current_profile['profile_picture_data'])
                col_pic, col_info = st.columns([1, 2])
                with col_pic:
                    st.image(img_data, width=150)
                with col_info:
                    st.markdown(f"### {current_profile['full_name']}")
                    st.caption(f"@{user['username']} • {user['role'].upper()}")
                    
                    if current_profile.get('job_title'):
                        st.write(f"**{current_profile['job_title']}**")
                    if current_profile.get('department'):
                        st.write(f"Department: {current_profile['department']}")
            except Exception as e:
                st.markdown(f"### {current_profile['full_name']}")
                st.caption(f"@{user['username']} • {user['role'].upper()}")
                
                if current_profile.get('job_title'):
                    st.write(f"**{current_profile['job_title']}**")
                if current_profile.get('department'):
                    st.write(f"Department: {current_profile['department']}")
        else:
            st.markdown(f"### {current_profile['full_name']}")
            st.caption(f"@{user['username']} • {user['role'].upper()}")
            
            if current_profile.get('job_title'):
                st.write(f"**{current_profile['job_title']}**")
            if current_profile.get('department'):
                st.write(f"Department: {current_profile['department']}")
        
        st.markdown("**Bio:**")
        st.info(current_profile.get('bio', 'No bio yet'))
        
        st.markdown("**Contact:**")
        st.write(f"Email: {current_profile.get('email', 'N/A')}")
        st.write(f"Phone: {current_profile.get('phone', 'Not provided') or 'Not provided'}")
        
        if current_profile.get('education'):
            st.markdown("**Education:**")
            st.info(current_profile['education'])
