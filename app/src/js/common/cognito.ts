import { CognitoAccessToken, CognitoAuthSession, CognitoIdToken, CognitoRefreshToken } from 'amazon-cognito-auth-js'

/**
 * Get auth status, refresh token if needed
 *
 * @returns {string} promise of the auth
 */
function getAuth (): string {
  const accessToken = localStorage.getItem('AccessToken') || ''
  const refreshToken = localStorage.getItem('RefreshToken') || ''
  const idToken = localStorage.getItem('IdToken') || ''
  const tokenType = localStorage.getItem('TokenType') || ''

  const exists = accessToken && refreshToken && idToken && tokenType
  if (!exists) {
    return ''
  }

  const sessionData = {
    IdToken: new CognitoIdToken(idToken),
    AccessToken: new CognitoAccessToken(accessToken),
    RefreshToken: new CognitoRefreshToken(refreshToken)
  }
  const cachedSession = new CognitoAuthSession(sessionData)
  let result = ''
  if (cachedSession.isValid()) {
    result = tokenType.concat(' ', accessToken)
  } else {
    const xhr = new XMLHttpRequest()
    const form = [
      ['grant_type', 'refresh_token'],
      ['client_id', localStorage.getItem('ClientID') || ''],
      ['refresh_token', refreshToken]
    ]
    xhr.open('POST',
      `https://${localStorage.getItem('URI')}/oauth2/token`, false)
    xhr.setRequestHeader('content-type', 'application/x-www-form-urlencoded')

    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4 && xhr.status === 200) {
        const data = JSON.parse(xhr.responseText)
        localStorage.setItem('AccessToken', data.access_token)
        localStorage.setItem('IdToken', data.id_token)
        localStorage.setItem('TokenType', data.token_type)
        result = '' + data.token_type + ' ' + data.access_token
      } else {
        throw new Error('Refresh token error')
      }
    }
    xhr.send(Array.from(
      form,
      (e) => e.map(encodeURIComponent).join('=')
    ).join('&'))
  }
  return result
}

export default {
  getAuth
}
