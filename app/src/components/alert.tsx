import React from "react"
import Alert, { Color } from "@material-ui/lab/Alert"
import IconButton from "@material-ui/core/IconButton"
import CloseIcon from "@material-ui/icons/Close"
import Session from "../common/session"
import { removeAlert } from "../action/common"
import { Severity } from "../types/common"

interface Props {
  id: string
  severity: Severity
  msg: string
  timeout: number
}

/** Custom window alert class */
export class CustomAlert extends React.Component<Props> {
  /** overrides default mounting */
  public componentDidMount(): void {
    setTimeout(() => {
      Session.dispatch(removeAlert(this.props.id))
    }, this.props.timeout)
  }

  /** overrides default render */
  public render(): React.ReactNode {
    return (
      <Alert
        severity={this.props.severity as Color}
        action={
          <IconButton
            aria-label="close"
            color="inherit"
            size="small"
            onClick={() => {
              Session.dispatch(removeAlert(this.props.id))
            }}
          >
            <CloseIcon fontSize="inherit" />
          </IconButton>
        }
      >
        {this.props.msg}
      </Alert>
    )
  }
}
