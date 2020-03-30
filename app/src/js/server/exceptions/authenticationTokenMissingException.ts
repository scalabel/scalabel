import HttpException from './httpException'

/**
 * Class represents authentication token missing exception
 * @extends HttpException
 */
class AuthenticationTokenMissingException extends HttpException {
  /**
   * Create an authentication token missing exception
   */
  constructor () {
    super(401, 'Authentication token missing')
  }
}

export default AuthenticationTokenMissingException
