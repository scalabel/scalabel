import { CognitoAuth } from 'amazon-cognito-auth-js'
import { CognitoUserPool, CognitoUserSession } from 'amazon-cognito-identity-js'
import React from 'react'

const appConfig = {
  region: 'us-west-2',
  userPool: 'us-west-2_tgxuoXSvV',
  userPoolBaseUri: 'https://scalabel.auth.us-west-2.amazoncognito.com',
  clientId: '52i44u3c7fapmec4oaqto4llc1',
  callbackUri: 'http://localhost:8686/callback',
  signoutUri: 'http://localhost:8686/signout',
  tokenScopes: [
    'openid',
    'email',
    'profile'
  ]
  // apiUri: 'https://localhost:8686'
}

/** Callback page */
class Callback extends React.Component {
  /**
   * component did mount
   */
  public componentDidMount () {
    this.decodeCode()
  }

  /**
   * public render method
   */
  public render () {
    return <div />
  }

  /**
   * decode code
   */
  private decodeCode = () => {
    this.parseCognitoWebResponse(window.location.href) // parse the callback URL
    .then(() => this.getCognitoSession()) // get a new session
    .then((token) => {
      sessionStorage.setItem('Authorization', token as string)
    })
    .catch()
  }

  /**
   * parse code config
   * @param {string} href - window location href
   */
  private parseCognitoWebResponse = (href: string) => {
    return new Promise((resolve, reject) => {
      const auth = this.createCognitoAuth()

      auth.userhandler = {
        onSuccess: (result) => {
          resolve(result)
        },
        onFailure: (err) => {
          reject(new Error('Failure parsing Cognito web response: ' + err))
        }
      }
      auth.parseCognitoWebResponse(href)
    })
  }

  /**
   * create auth
   */
  private createCognitoAuth = () => {
    const appWebDomain =
      appConfig.userPoolBaseUri.replace('https://', '').replace('http://', '')
    const auth = new CognitoAuth({
      UserPoolId: appConfig.userPool,
      ClientId: appConfig.clientId,
      AppWebDomain: appWebDomain,
      TokenScopesArray: appConfig.tokenScopes,
      RedirectUriSignIn: appConfig.callbackUri,
      RedirectUriSignOut: appConfig.signoutUri
    })
    return auth
  }

  /**
   * get session
   */
  private getCognitoSession = () => {
    return new Promise((resolve, reject) => {
      const cognitoUser = this.createCognitoUser()
      if (cognitoUser) {
        cognitoUser.getSession((err: Error, result: CognitoUserSession) => {
          if (err || !result) {
            reject(new Error('Failure getting Cognito session: ' + err))
            return
          }

          resolve(result.getAccessToken().getJwtToken())
        })
      }
    })
  }

  /**
   * create user
   */
  private createCognitoUser = () => {
    const pool = this.createCognitoUserPool()
    return pool.getCurrentUser()
  }

  /**
   * create user pool
   */
  private createCognitoUserPool = () => new CognitoUserPool({
    UserPoolId: appConfig.userPool,
    ClientId: appConfig.clientId
  })
}

export default Callback
