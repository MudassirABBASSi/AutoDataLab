# 🔐 AutoDataLab Authentication System Guide

## Overview

I've created a complete authentication system for your AutoDataLab application with the following features:

- ✅ User login and registration
- ✅ Password hashing (SHA-256)
- ✅ Session management
- ✅ User roles (Admin/User)
- ✅ User profile display
- ✅ Password change functionality
- ✅ User management (Admin only)
- ✅ Secure logout
- ✅ Beautiful, professional UI

## 📁 Files Created

### 1. `core/authentication.py`
The main authentication module containing:
- `AuthManager` - Handles user authentication and credential management
- `login_page()` - Beautiful login/registration UI
- `show_user_profile()` - Displays user info in sidebar
- `user_management_page()` - Admin interface for managing users
- Authentication helper functions

### 2. `app_with_auth.py`
Example implementation showing how to integrate authentication into your existing app.

### 3. `users.json` (auto-created)
Stores user credentials securely (passwords are hashed).

## 🚀 Quick Start

### Option 1: Test the Authentication System (Recommended First)

Run the example app to see authentication in action:

```powershell
cd D:\AutoDataLab-main\AutoDataLab-main
& D:\AutoDataLab-main\.venv\Scripts\python.exe -m streamlit run app_with_auth.py
```

**Default credentials:**
- Username: `admin`
- Password: `admin123`

### Option 2: Integrate into Your Existing App

To add authentication to your existing `app.py`, add these lines:

#### Step 1: Import authentication modules

Add to your imports at the top of `app.py`:

```python
from core import (
    # ... your existing imports ...
    
    # Add these authentication imports
    init_session_state,
    is_authenticated,
    get_current_user,
    login_page,
    show_user_profile,
    user_management_page,
)
```

#### Step 2: Initialize session state

Right after `st.set_page_config()`, add:

```python
# Initialize authentication session state
init_session_state()
```

#### Step 3: Add authentication check

Before your main app content, add:

```python
# Check authentication
if not is_authenticated():
    login_page()
    st.stop()  # Prevents rest of app from loading

# Show user profile in sidebar
show_user_profile()
```

#### Step 4: (Optional) Add user management page

In your sidebar navigation, add:

```python
# Get current user
user = get_current_user()

# Add to navigation menu
if user['role'] == 'admin':
    page = st.radio("Navigation", ["Data Analysis", "User Management"])
    
    if page == "User Management":
        user_management_page()
```

## 📋 Complete Integration Example

Here's a minimal example of your `app.py` with authentication:

```python
import streamlit as st
from core import (
    # Your existing imports
    init_session_state, is_authenticated, 
    get_current_user, login_page, show_user_profile
)

# Configure page
st.set_page_config(page_title="AutoDataLab", layout="wide")

# Initialize auth
init_session_state()

# Authentication gate
if not is_authenticated():
    login_page()
    st.stop()

# Show user profile
show_user_profile()

# Get current user
user = get_current_user()

# Your app content
st.title("🔬 AutoDataLab")
st.write(f"Welcome, **{user['full_name']}**!")

# Rest of your app...
```

## 🎨 Features

### 1. Login Page
- Professional design with tabs for Login/Register
- Form validation
- Error handling
- Default admin credentials shown

### 2. User Registration
- Username, full name, email, password fields
- Password confirmation
- Minimum password length validation
- Duplicate username detection

### 3. User Profile (Sidebar)
- Displays user's full name
- Shows role and email
- Login timestamp
- Logout button

### 4. User Management (Admin Only)
- View all registered users
- Delete users (except admin)
- Change password
- User role display

### 5. Security Features
- **Password hashing**: SHA-256 encryption
- **Session management**: Streamlit session state
- **Role-based access**: Admin vs. User permissions
- **Secure storage**: Passwords never stored in plain text

## 👥 User Roles

### Admin Role
- Full access to all features
- Can manage other users
- Can view user list
- Cannot be deleted

### User Role
- Access to main application features
- Cannot access user management
- Can change own password
- Can be deleted by admin

## 🔧 Customization

### Change Password Requirements

Edit in `core/authentication.py`:

```python
def register_user(self, username: str, password: str, ...):
    if len(password) < 6:  # Change minimum length here
        return False, "Password must be at least 6 characters"
```

### Modify Default Admin Credentials

Edit the `_ensure_users_file()` method in `AuthManager`:

```python
default_users = {
    "admin": {
        "password": self._hash_password("YOUR_NEW_PASSWORD"),
        "role": "admin",
        "email": "YOUR_EMAIL@example.com",
        "full_name": "YOUR_NAME"
    }
}
```

### Add More User Fields

In `core/authentication.py`, modify the user dictionary structure:

```python
users[username] = {
    'password': self._hash_password(password),
    'role': role,
    'email': email,
    'full_name': full_name,
    'created_at': datetime.now().isoformat(),
    'phone': phone,  # Add new field
    'department': department,  # Add new field
}
```

### Customize Login Page Styling

Modify the CSS in the `login_page()` function:

```python
st.markdown("""
    <style>
    .login-container {
        max-width: 450px;
        /* Modify these styles */
    }
    </style>
""", unsafe_allow_html=True)
```

## 📊 User Data Storage

User data is stored in `users.json` in this format:

```json
{
    "admin": {
        "password": "hashed_password_here",
        "role": "admin",
        "email": "admin@autodatalab.com",
        "created_at": "2026-02-27T10:30:00",
        "full_name": "Administrator"
    },
    "john_doe": {
        "password": "hashed_password_here",
        "role": "user",
        "email": "john@example.com",
        "created_at": "2026-02-27T11:00:00",
        "full_name": "John Doe"
    }
}
```

## 🛡️ Security Best Practices

### Current Implementation
✅ SHA-256 password hashing
✅ Session-based authentication
✅ Role-based access control
✅ Input validation

### For Production (Future Enhancements)
- 🔄 Use bcrypt instead of SHA-256 for password hashing
- 🔄 Add session timeout (30 minutes of inactivity)
- 🔄 Implement "Remember Me" functionality
- 🔄 Add email verification
- 🔄 Implement password reset via email
- 🔄 Add login attempt tracking
- 🔄 Use database (SQLite/PostgreSQL) instead of JSON
- 🔄 Add 2-factor authentication (2FA)
- 🔄 Log authentication events

## 🧪 Testing the System

### Test Login
1. Run the app
2. Use credentials: `admin` / `admin123`
3. Verify successful login

### Test Registration
1. Click "Register" tab
2. Fill in all fields
3. Create new account
4. Login with new credentials

### Test User Management (Admin)
1. Login as admin
2. Navigate to "User Management"
3. View registered users
4. Try changing password
5. Delete a test user

### Test Access Control
1. Create a regular user account
2. Login as regular user
3. Verify no "User Management" option appears

## 📝 API Reference

### `is_authenticated() -> bool`
Returns True if user is logged in.

### `get_current_user() -> Dict`
Returns current user info dictionary:
```python
{
    'username': 'john_doe',
    'role': 'user',
    'email': 'john@example.com',
    'full_name': 'John Doe'
}
```

### `logout()`
Logs out current user and clears session.

### `login_page()`
Displays login/registration interface.

### `show_user_profile()`
Displays user profile card in sidebar.

### `user_management_page()`
Displays admin user management interface.

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'authentication'"

**Solution**: Make sure you've updated `/core/__init__.py` to include authentication imports.

### Issue: Users.json not found

**Solution**: The file is created automatically on first run. Ensure write permissions in app directory.

### Issue: Can't logout

**Solution**: Click the logout button in the sidebar. Session state will be cleared.

### Issue: Forgot admin password

**Solution**: Delete `users.json` file. It will be recreated with default credentials on next run.

## 🎯 Next Steps

1. **Test the system**: Run `app_with_auth.py` to see it in action
2. **Integrate into your app**: Follow the integration steps above
3. **Customize**: Modify styles, add fields, adjust requirements
4. **Enhance security**: Consider production enhancements listed above
5. **Add features**: Session timeout, password reset, email verification

## 📞 Support

If you need help with:
- Integration into specific parts of your app
- Custom authentication logic
- Database migration from JSON
- Additional security features

Just let me know what you need!

---

**Happy coding! 🚀**
