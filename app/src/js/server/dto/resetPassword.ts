import { IsNotEmpty, IsString } from 'class-validator'

/**
 * Represents forget password dto
 */
class ResetPasswordDto {
  /**
   * password
   */
  @IsString()
  @IsNotEmpty()
  public password: string = ''
  /**
   * token
   */
  @IsString()
  @IsNotEmpty()
  public token: string = ''
}

export default ResetPasswordDto
