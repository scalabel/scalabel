import { AuthenticationDetails, CognitoUser, CognitoUserAttribute, CognitoUserPool, CognitoUserSession } from 'amazon-cognito-identity-js'
import fetch, { RequestInfo, RequestInit, Response } from 'node-fetch'
import CreateUserDto from '../dto/createUser'
import ForgetPasswordDto from '../dto/forgetPassword'
import LogInDto from '../dto/login'
import ResetPasswordDto from '../dto/resetPassword'
import { User } from '../entities/user'
import { ServerConfig } from '../types'
import { AuthenticationImpl } from './auth'

import Global = NodeJS.Global
export interface GlobalWithCognitoFix extends Global {
  /** fetch interface */
  fetch: (url: RequestInfo,
          init?: RequestInit
         ) => Promise<Response>
}
declare const global: GlobalWithCognitoFix
global.fetch = fetch

/**
 * Authentication service with AWS cognito implementation
 */
class AuthenticationCognitoService implements AuthenticationImpl {

  /** cognito user pool entity */
  private userPool: CognitoUserPool

  constructor (config: ServerConfig) {
    if (!config.poolId || !config.clientId) {
      throw new Error('Cognito config error')
    }
    this.userPool = new CognitoUserPool({
      UserPoolId: config.poolId,
      ClientId: config.clientId
    })
  }

  /**
   * Create user
   * @param {CreateUserDto} registerData - Create user dto
   * @param {AuthenticationImpl~registerCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public register = (registerData: CreateUserDto,
                     _success: (user: User) => void,
                     error: (error: Error) => void) => {
    const attributeList = [
      new CognitoUserAttribute({
        Name: 'given_name',
        Value: registerData.givenName
      }),
      new CognitoUserAttribute({
        Name: 'family_name',
        Value: registerData.familyName
      }),
      new CognitoUserAttribute({
        Name: 'custom:created_at',
        Value: new Date().getTime().toString()
      }),
      new CognitoUserAttribute({
        Name: 'updated_at',
        Value: new Date().getTime().toString()
      })
    ]
    this.userPool.signUp(
      registerData.email, registerData.password, attributeList, [],
      (err, result) => {
        if (err) {
          error(err)
        }
        if (result) {
          const cognitoUser = result.user
          cognitoUser.getSession(
            (_cogerr: Error, _session: CognitoUserSession) => {
              // TODO: do something
            })
        }
      }
    )
  }

  /**
   * Login
   * @param logInData login data
   * @param {AuthenticationImpl~loginCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public login = (logInData: LogInDto,
                  _success: (user: User) => void,
                  error: (error: Error) => void) => {
    const user = new CognitoUser({
      Username: logInData.email,
      Pool: this.userPool
    })
    user.authenticateUser(new AuthenticationDetails({
      Username: logInData.email,
      Password: logInData.password
    }), {
      onSuccess: (_result) => {
        // TODO: do something
      },
      onFailure: (err) => {
        error(err)
      }
    })
  }

  /**
   * Reset password
   * @param {ResetPasswordDto} resetData - Reset password data
   * @param {AuthenticationImpl~resetCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public resetPassword = (_resetData: ResetPasswordDto) => {
    throw new Error('Method not implemented.')
  }

  /**
   * Forget password logic handling
   * @param {ForgetPasswordDto} forgetPasswordDto - forget password data
   * @param {AuthenticationImpl~forgetCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public forgetPassword = (_forgetPasswordDto: ForgetPasswordDto) => {
    throw new Error('Method not implemented.')
  }

}

export default AuthenticationCognitoService
