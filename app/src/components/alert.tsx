import React from "react"
import Alert, { Color } from "@material-ui/lab/Alert"
import AlertTitle from "@material-ui/lab/AlertTitle"
import IconButton from "@material-ui/core/IconButton"
import CloseIcon from "@material-ui/icons/Close"
import Session from "../common/session"
import { closeAlert } from "../action/common"
import { Severity } from "../types/common"

interface Props {
  id: string
  severity: Severity
  msg: string
}

/** Custom window alert class */
export class CustomAlert extends React.Component<Props> {
  /** overrides default mounting */
  public componentDidMount(): void {
    setTimeout(() => {
      Session.dispatch(closeAlert(this.props.id))
    }, 8000)
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
              Session.dispatch(closeAlert(this.props.id))
            }}
          >
            <CloseIcon fontSize="inherit" />
          </IconButton>
        }
      >
        <AlertTitle>
          <b>{this.props.severity.toUpperCase()}</b>
        </AlertTitle>
        {this.props.msg}
      </Alert>
    )
  }
}
