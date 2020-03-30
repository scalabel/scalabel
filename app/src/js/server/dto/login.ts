import { IsEmail, IsNotEmpty, IsString } from 'class-validator'

/**
 * Represents login dto
 */
class LogInDto {
  /**
   * email
   */
  @IsEmail()
  public email: string = ''

  /**
   * password
   */
  @IsString()
  @IsNotEmpty()
  public password: string = ''
}

export default LogInDto
