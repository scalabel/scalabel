/**
 * Class represents HTTP exception
 *
 * @augments Error
 */
class HttpException extends Error {
  /** Status code */
  public status: number
  /** Error message */
  public message: string
  /**
   * Create a HTTP Exception
   *
   * @param {number} status - Error code
   * @param {string} message - Error message
   */
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.message = message
  }
}

export default HttpException
