import HttpException from './httpException'

/**
 * Class represents wrong authentication token exception
 * @extends HttpException
 */
class WrongAuthenticationTokenException extends HttpException {
  /**
   * Create a wrong authentication token exception
   */
  constructor () {
    super(401, 'Wrong authentication token')
  }
}

export default WrongAuthenticationTokenException
