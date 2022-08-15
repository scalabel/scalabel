import { uid } from "../common/uid"
import Session from "../common/session"
import { Severity } from "../types/common"
import { addAlert } from "../action/common"

/**
 * Custom window alert
 *
 * @param severity - severity of the alert
 * @param msg - message to display
 */
export function alert(severity: Severity, msg: string): void {
  const alert = {
    id: uid(),
    severity,
    message: msg,
    timeout: 12000
  }
  Session.dispatch(addAlert(alert))
}
