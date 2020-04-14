import CreateUserDto from '../dto/createUser'
import ForgetPasswordDto from '../dto/forgetPassword'
import LogInDto from '../dto/login'
import ResetPasswordDto from '../dto/resetPassword'
import { Reset } from '../entities/reset'
import { User } from '../entities/user'
import { ServerConfig } from '../types'
import AuthenticationCognitoService from './authCognitoService'
import AuthenticationLocalService from './authLocalService'

export interface AuthenticationImpl {
  /**
   * Create user
   * @param {CreateUserDto} registerData - Create user dto
   * @param {AuthenticationImpl~registerCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  register (registerData: CreateUserDto,
            success: (user: User) => void,
            error: (error: Error) => void): void
  /**
   * Login
   * @param logInData login data
   * @param {AuthenticationImpl~loginCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  login (logInData: LogInDto,
         success: (user: User) => void,
         error: (error: Error) => void): void
  /**
   * Reset password
   * @param {ResetPasswordDto} resetData - Reset password data
   * @param {AuthenticationImpl~resetCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  resetPassword (resetData: ResetPasswordDto,
                 success: () => void,
                 error: (error: Error) => void): void
  /**
   * Forget password logic handling
   * @param {ForgetPasswordDto} forgetPasswordDto - forget password data
   * @param {AuthenticationImpl~forgetCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  forgetPassword (forgetPasswordDto: ForgetPasswordDto,
                  success: (reset: Reset) => void,
                  error: (error: Error) => void): void
}

/**
 * Authentication service
 */
class AuthenticationService {

  /** Implementation */
  private impl: AuthenticationImpl

  /**
   * Create an authentication controller
   * @param {ServerConfig} config - Server config
   */
  constructor (config: ServerConfig) {
    if (config.authProvider === 'cognito') {
      this.impl = new AuthenticationCognitoService(config)
    } else {
      this.impl = new AuthenticationLocalService(config)
    }
  }

  /**
   * Create user
   * @param {CreateUserDto} registerData - Create user dto
   */
  public register = async (registerData: CreateUserDto) => {
    return new Promise((resolve, reject) => {
      this.impl.register(registerData,
        (user) => resolve(user), (error) => reject(error))
    })
  }

  /**
   * Login
   * @param logInData login data
   */
  public login = (logInData: LogInDto) => {
    return new Promise((resolve, reject) => {
      this.impl.login(logInData,
        (user) => resolve(user), (error) => reject(error))
    })
  }

  /**
   * Reset password
   * @param {ResetPasswordDto} resetData - Reset password data
   */
  public resetPassword = (resetData: ResetPasswordDto) => {
    return new Promise((resolve, reject) => {
      this.impl.resetPassword(resetData,
        () => resolve(200), (error) => reject(error))
    })
  }

  /**
   * Forget password logic handling
   * @param {ForgetPasswordDto} forgetPasswordDto - forget password data
   */
  public forgetPassword =
  (forgetPasswordDto: ForgetPasswordDto): Promise<Reset> => {
    return new Promise((resolve, reject) => {
      this.impl.forgetPassword(forgetPasswordDto,
        (reset) => resolve(reset), (error) => reject(error))
    })
  }
}

export default AuthenticationService
