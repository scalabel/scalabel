/**
 * Defining the types of some general callback functions
 */
export type MaybeError = Error | undefined

/**
 * Severity of alert
 */
export enum Severity {
  SUCCESS = "success",
  ERROR = "error",
  WARNING = "warning",
  INFO = "info"
}
