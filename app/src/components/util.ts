import { SubmitData } from "../types/state"

/**
 * Puts Date.now into dashboard display format
 *
 * @param dateNow
 */
export function formatDate(dateNow: number): string {
  const date = new Date(dateNow)
  return date.toLocaleString("en-CA", { hour12: false })
}

/**
 * Extract timestamp of latest submission; -1 if not submitted
 *
 * @param submissions
 */
export function getSubmissionTime(submissions: SubmitData[]): number {
  if (submissions.length > 0) {
    const latestSubmission = submissions[submissions.length - 1]
    return latestSubmission.time
  }
  return -1
}
