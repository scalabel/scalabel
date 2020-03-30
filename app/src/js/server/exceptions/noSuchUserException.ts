import HttpException from './httpException'

/**
 * Class represents no user exception
 * @extends HttpException
 */
class NoSuchUserException extends HttpException {
  /**
   * Create a not such user exception
   */
  constructor () {
    super(404, 'No such user')
  }
}

export default NoSuchUserException
