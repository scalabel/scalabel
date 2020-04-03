import HttpException from './httpException'

/**
 * Class represents server side unknown error
 */
class ResetTokenExpiredException extends HttpException {
  /**
   * Create a server error exception
   */
  constructor () {
    super(400, 'Reset token expired')
  }
}

export default ResetTokenExpiredException
