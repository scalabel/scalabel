import HttpException from './httpException'

/**
 * Class represents credential exception
 * @extends HttpException
 */
class WrongCredentialsException extends HttpException {
  /**
   * Create a wrong credential exception
   */
  constructor () {
    super(401, 'Wrong credentials provided')
  }
}

export default WrongCredentialsException
