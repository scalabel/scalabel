import HttpException from './httpException'

/**
 * Class represents server side unknown error
 */
class ServerErrorException extends HttpException {
  /**
   * Create a server error exception
   */
  constructor () {
    super(500, 'Server error')
  }
}

export default ServerErrorException
