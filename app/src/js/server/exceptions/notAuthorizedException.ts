import HttpException from './httpException'

/**
 * Class represents not authorized exception
 * @extends HttpException
 */
class NotAuthorizedException extends HttpException {
  /**
   * Create a not authorized exception
   */
  constructor () {
    super(403, 'Not authorized')
  }
}

export default NotAuthorizedException
