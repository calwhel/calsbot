# TradeHub — App Store Data Safety & Privacy Documentation

This document provides the answers needed for Apple App Store Privacy Nutrition
Labels and Google Play Data Safety forms.

---

## 1. Data Types Collected

| Data Type | Collected | Purpose | Linked to Identity | Shared with Third Parties |
|---|---|---|---|---|
| Email address | Yes | Account creation, authentication, support | Yes | No |
| Name / Display name | Optional | Personalisation | Yes | No |
| User ID (UID) | Yes | Authentication, session management | Yes | No |
| Apple ID (sub claim) | Yes (iOS) | Sign in with Apple | Yes | No |
| Password (hashed) | Yes (email auth) | Authentication | No (hashed) | No |
| Exchange API keys | Yes | Trade execution on user's behalf | Yes | Yes (to user's chosen exchange only) |
| Push notification token | Yes | Deliver trade alerts | Yes | Yes (Expo Push Service) |
| Device type / OS | Yes | Platform-specific behaviour | No | No |
| Trading activity | Yes | Strategy execution, analytics | Yes | No |
| Purchase history | Yes (IAP) | Subscription entitlements | Yes | Yes (Apple/Google/RevenueCat) |

## 2. Data NOT Collected

- Precise location / GPS
- Contacts / Address book
- Photos / Camera / Microphone
- Browsing history
- Health or fitness data
- Financial information beyond exchange API keys
- Advertising identifiers
- Device sensor data

## 3. Data Usage Purposes

| Purpose | Details |
|---|---|
| App functionality | Core trading, strategy management, backtesting |
| Authentication | Email/password, UID, Sign in with Apple |
| Push notifications | Trade alerts, strategy signals, P&L updates |
| Subscription management | In-app purchase verification via RevenueCat |
| Analytics (first-party only) | Aggregate usage to improve features — no third-party analytics SDKs |

## 4. Data Sharing

| Recipient | Data Shared | Purpose |
|---|---|---|
| Cryptocurrency exchanges (Bybit, Binance, etc.) | API keys, trade orders | Execute trades on user's behalf |
| Apple / Google Play | Purchase receipts | Process in-app subscriptions |
| RevenueCat | Anonymous user ID, purchase receipts | Manage cross-platform subscription entitlements |
| Expo | Push tokens | Deliver push notifications |
| AI providers (OpenAI, Anthropic, Google) | Strategy parameters (no PII) | AI-powered strategy suggestions |

No data is shared with advertisers or data brokers.

## 5. Data Retention

- **Active accounts**: Data retained while account is active.
- **Deleted accounts**: PII is cleared within 30 days of account deletion. Anonymised trade history may be retained for analytical purposes.
- **Push tokens**: Removed immediately upon sign-out or account deletion.

## 6. Security Practices

- All data transmitted via TLS/HTTPS.
- Exchange API keys stored encrypted at rest.
- Passwords hashed with bcrypt.
- Mobile credentials stored in iOS Keychain / Android Keystore via expo-secure-store.
- Account deletion available in-app (Settings > Delete account).

## 7. Account Deletion

Users can delete their account from the mobile app:
**Settings > Delete account**

This action:
- Deactivates all trading strategies
- Removes all push notification tokens
- Clears personal data (email, name, password hash, Apple ID)
- Marks the account as permanently deleted

Deletion is processed immediately. Users can also request deletion by emailing
hi@tradehub.markets.

## 8. Children's Privacy

TradeHub is not directed at children under 18. We do not knowingly collect
information from minors.

## 9. Apple App Store Privacy Nutrition Labels

### Data Used to Track You
None — TradeHub does not track users across other apps or websites.

### Data Linked to You
- Contact info (email address)
- Identifiers (user ID)
- Purchases (subscription status)

### Data Not Linked to You
- Usage data (aggregated, not linked to identity)
- Diagnostics (crash logs, if any)

## 10. Google Play Data Safety Answers

- **Does your app collect or share any of the required user data types?** Yes
- **Is all of the user data collected by your app encrypted in transit?** Yes
- **Do you provide a way for users to request that their data is deleted?** Yes (in-app and via email)
- **Data types collected**: Email, User ID, Purchase history, Push tokens
- **Data types shared**: Purchase history (with Apple/Google/RevenueCat for subscription management)

## 11. Contact

Privacy inquiries: hi@tradehub.markets
Privacy policy: https://tradehubmarkets.com/privacy
Terms of service: https://tradehubmarkets.com/terms
