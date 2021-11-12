import React from "react"
import Alert from "@mui/material/Alert"
import IconButton from "@mui/material/IconButton"
import CloseIcon from "@mui/icons-material/Close"
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
        severity={this.props.severity}
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
