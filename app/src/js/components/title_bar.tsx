import * as fa from '@fortawesome/free-solid-svg-icons/index'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { AppBar, IconButton, Toolbar, Tooltip } from '@material-ui/core'
import Fade from '@material-ui/core/Fade'
import { Theme } from '@material-ui/core/styles'
import createStyles from '@material-ui/core/styles/createStyles'
import { withStyles } from '@material-ui/core/styles/index'
import Typography from '@material-ui/core/Typography'
import _ from 'lodash'
import React from 'react'
import { submit } from '../action/common'
import Session, { ConnectionStatus } from '../common/session'
import Synchronizer from '../common/synchronizer'
import { Key } from '../common/types'
import { defaultAppBar } from '../styles/general'
import { StatusMessageBox } from '../styles/label'
import { Component } from './component'

// how long to wait until saving times out
export const saveTimeout = 20000

const styles = (theme: Theme) => createStyles({
  appBar: {
    ...defaultAppBar,
    position: 'relative',
    height: '100%'
  },
  grow: {
    flexGrow: 1
  },
  titleUnit: {
    color: '#bbbbbb',
    margin: theme.spacing(0) * 0.5
  }
})

interface ClassType {
  /** app bar class */
  appBar: string,
  /** grow class for spacing */
  grow: string,
  /** title unit class */
  titleUnit: string
}

interface Props {
  /** Styles of TitleBar */
  classes: ClassType
  /** Theme of TitleBar */
  theme: Theme
  /** title of TitleBar */
  title: string
  /** dashboardLink of TitleBar */
  dashboardLink: string
  /** instructionLink of TitleBar */
  instructionLink: string
  /** whether to show save button or to autosave */
  autosave: boolean
  /** synchronizer */
  synchronizer: Synchronizer
}

// /**
//  * Save the current state to the server
//  */
// function save () {
//   Session.updateStatusDisplay(ConnectionStatus.SAVING)
//   const state = Session.getState()
//   const xhr = new XMLHttpRequest()
//   xhr.timeout = saveTimeout
//   xhr.onreadystatechange = () => {
//     if (xhr.readyState === 4) {
//       if (JSON.parse(xhr.response) !== 0) {
//         alert('Save failed.')
//         Session.updateStatusDisplay(ConnectionStatus.UNSAVED)
//       } else {
//         Session.updateStatusDisplay(ConnectionStatus.SAVED)
//         setTimeout(() => {
//           Session.updateStatusDisplay(ConnectionStatus.UNSAVED)
//         }, 5000)
//       }
//     }
//   }
//   xhr.open('POST', './postSaveV2')
//   xhr.send(JSON.stringify(state))
// }

/**
 * Title bar
 */
class TitleBar extends Component<Props> {
  /** Listener for key down events */
  private _keyDownListener: (e: KeyboardEvent) => void

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: Props) {
    super(props)
    // Update the StatusMessageBox when the Session status changes
    Session.applyStatusEffects = () => {
      this.forceUpdate()
    }
    this._keyDownListener = ((e: KeyboardEvent) => {
      if (e.key === Key.S_LOW || e.key === Key.S_UP) {
        this.save()
      }
    })
  }

  /** Mount override */
  public componentDidMount () {
    document.addEventListener('keydown', this._keyDownListener)
  }

  /**
   * Unmount
   * Disables asynchronous callbacks
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    // De-couple the titlebar and the session
    Session.applyStatusEffects = () => { return }
    document.removeEventListener('keydown', this._keyDownListener)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    const { title } = this.props
    const { instructionLink } = this.props
    const { dashboardLink } = this.props
    const { autosave } = this.props

    const buttonInfo: Array<{
      /** Name */
      title: string,
      /** Icon */
      icon: fa.IconDefinition,
      /** Link */
      href?: string,
      /** Listener  */
      onClick?: () => void
    }> = [
      { title: 'Instructions', href: instructionLink, icon: fa.faInfo },
      { title: 'Keyboard Usage', icon: fa.faQuestion },
      { title: 'Dashboard', href: dashboardLink, icon: fa.faList }
    ]
    // if autosave is on, don't need manual save button
    let submitHandler = () => { Session.dispatch(submit()) }
    if (!autosave) {
      submitHandler = () => {
        const submitAction = submit()
        Session.dispatch(submitAction)
        // save after, so submit flag is also saved
        this.save()
      }
    }
    if (!autosave) {
      buttonInfo.push(
        { title: 'Save', onClick: () => { this.save() }, icon: fa.faSave })
    }
    buttonInfo.push(
      { title: 'Submit', onClick: submitHandler, icon: fa.faCheck })

    const buttons = buttonInfo.map((b) => {
      const onClick = _.get(b, 'onClick', undefined)
      const href = _.get(b, 'href', '#')
      const target = ('href' in b ? 'view_window' : '_self')
      return (
              <Tooltip title={b.title} key={b.title}>
                <IconButton className={classes.titleUnit} onClick={onClick}
                            href={href} target={target} data-testid={b.title}>
                  <FontAwesomeIcon icon={b.icon} size='xs'/>
                </IconButton>
              </Tooltip>
      )
    })

    let sessionStatus: string
    let hideMessage = false
    switch (Session.status) {
      case ConnectionStatus.SAVING:
      case ConnectionStatus.NOTIFY_SAVED: {
        sessionStatus = this.getTextForStatus(Session.status)
        if (autosave) {
          hideMessage = true
        }
        break
      }
      case ConnectionStatus.RECONNECTING: {
        sessionStatus = this.getTextForStatus(Session.status)
        break
      }
      case ConnectionStatus.SAVED:
      case ConnectionStatus.UNSAVED: {
        // Want the text fade animation
        sessionStatus = this.getTextForStatus(Session.prevStatus)
        hideMessage = true
        break
      }
      default: {
        sessionStatus = 'Error occured, try refreshing.'
        break
      }
    }

    return (
      <AppBar className={classes.appBar}>
        <Toolbar>
          <Typography variant='h6' noWrap>
            {title}
          </Typography>
          <Fade in={!hideMessage} timeout={300}>
            <StatusMessageBox>
              {sessionStatus}
            </StatusMessageBox>
          </Fade>
          <div className={classes.grow}/>
          {buttons}
        </Toolbar>
      </AppBar>
    )
  }

   /** Select text based on connection status */
  private getTextForStatus (status: ConnectionStatus) {
    switch (status) {
      case ConnectionStatus.SAVING: {
        return 'Saving in progress...'
      }
      case ConnectionStatus.RECONNECTING: {
        return 'Trying to reconnect...'
      }
      case ConnectionStatus.NOTIFY_SAVED:
      default: {
        return 'All progress saved.'
      }
    }
  }

  /** Save task */
  private save () {
    this.props.synchronizer.sendQueuedActions()
  }
}

export default withStyles(styles, { withTheme: true })(TitleBar)
