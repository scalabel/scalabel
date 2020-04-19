import { CognitoAccessToken, CognitoAuthSession, CognitoIdToken, CognitoRefreshToken } from 'amazon-cognito-auth-js'
import { Endpoint } from '../server/types'

/**
 * This function post request to backend to retrieve required data to display
 * @param {string} url
 * @param {string} method
 * @param {boolean} async
 * @return {function} data
 */
export function requestData<DataType> (url: string,
                                       method: string,
                                       async: boolean): DataType[] {
  let data!: DataType[]
  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        data = JSON.parse(xhr.responseText)
      } else if (xhr.status === 401) {
        const res = JSON.parse(xhr.responseText)
        window.location.href = res.redirect
      }
    }
  }
  xhr.open(method, url, async)
  const auth = getAuth()
  if (auth) {
    xhr.setRequestHeader('Authorization', auth)
  }
  xhr.send(null)
  return data
}

/**
 * Get auth status, refresh token if needed
 *
 * @export
 * @returns {string} promise of the auth
 */
export function getAuth (): string {
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

/**
 * This function get request to backend to retrieve users' information
 * @return {string[]} projects
 */
export function getProjects (): string[] {
  return requestData(Endpoint.GET_PROJECT_NAMES, 'get', false)
}

/**
 * Redirect user to create new projects
 */
export function goCreate (): void {
  window.location.href = '/create'
}

/**
 * Redirect user to logOut page
 */
export function logout (): void {
  window.location.href = '/logOut'
}

/**
 * Redirect user(either admin or worker) to the project's dashboard
 * @param {string} projectName - the values to convert.
 */
export function toProject (projectName: string): void {
  window.location.href = '/dashboard?project_name=' + projectName
}
