// tslint:disable:no-any
// TODO: remove the disable tag
import * as fa from '@fortawesome/free-solid-svg-icons/index'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { AppBar, Box, IconButton, Toolbar, Tooltip } from '@material-ui/core'
import createStyles from '@material-ui/core/styles/createStyles'
import { withStyles } from '@material-ui/core/styles/index'
import Typography from '@material-ui/core/Typography'
import _ from 'lodash'
import React from 'react'
import Session from '../common/session'
import { ConnectionStatus } from '../functional/types'
import { defaultAppBar } from '../styles/general'
import { Component } from './component'

const styles: any = (theme: any) => createStyles({
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
    margin: theme.spacing.unit * 0.5
  }
})

interface Props {
  /** Styles of TitleBar */
  classes: any
  /** Theme of TitleBar */
  theme: any
  /** title of TitleBar */
  title: string
  /** dashboardLink of TitleBar */
  dashboardLink: string
  /** instructionLink of TitleBar */
  instructionLink: string
}

/**
 * Save the current state to the server
 */
function save (callerComponent: TitleBar) {
  Session.status = ConnectionStatus.SAVING
  callerComponent.forceUpdate()
  const state = Session.getState()
  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      Session.status = ConnectionStatus.SAVED
      callerComponent.forceUpdate()
      setTimeout(() => {
        Session.status = ConnectionStatus.UNSAVED
        callerComponent.forceUpdate()
      }, 5000)
      if (JSON.parse(xhr.response) !== 0) {
        alert('Save failed.')
      }
    }
  }
  xhr.open('POST', './postSaveV2')
  xhr.send(JSON.stringify(state))
}

/**
 * turn assistant view on/off
 */
function toggleAssistantView () {
  // Session.dispatch({type: types.TOGGLE_ASSISTANT_VIEW});
}

/**
 * Title bar
 */
class TitleBar extends Component<Props> {
  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: any) {
    super(props)
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

    const buttonInfo = [
      { title: 'Instructions', href: instructionLink, icon: fa.faInfo },
      { title: 'Keyboard Usage', icon: fa.faQuestion },
      { title: 'Dashboard', href: dashboardLink, icon: fa.faList },
      {
        title: 'Assistant View', onClick: toggleAssistantView,
        icon: fa.faColumns
      },
      { title: 'Save', onClick: () => { save(this) }, icon: fa.faSave },
      { title: 'Submit', onClick: () => { save(this) }, icon: fa.faCheck }
    ]
    const buttons = buttonInfo.map((b) => {
      const onClick = _.get(b, 'onClick', undefined)
      const href = _.get(b, 'href', '#')
      const target = ('href' in b ? 'view_window' : '_self')
      return (
              <Tooltip title={b.title}>
                <IconButton className={classes.titleUnit} onClick={onClick}
                            href={href} target={target}>
                  <FontAwesomeIcon icon={b.icon} size='xs'/>
                </IconButton>
              </Tooltip>
      )
    })

    let sessionStatus: string
    let hideMessage = false
    switch (Session.status) {
      case ConnectionStatus.SAVING: {
        sessionStatus = 'Saving in progress...'
        break
      }
      case ConnectionStatus.SAVED: {
        sessionStatus = 'All progress saved.'
        break
      }
      case ConnectionStatus.RECONNECTING: {
        sessionStatus = 'Trying to reconnect...'
        break
      }
      case ConnectionStatus.UNSAVED: {
        sessionStatus = ''
        hideMessage = true
        break
      }
      default: {
        sessionStatus = 'Error occured, try refreshing.'
        break
      }
    }

    const statusMessage = hideMessage ? null : (
      <Box marginLeft={5} borderRadius='borderRadius'
        color='text.primary' bgcolor='#909090'
        padding='3px' fontSize='fontSize'>
        {sessionStatus}
      </Box>
    )

    return (
            <AppBar className={classes.appBar}>
              <Toolbar>
                <Typography variant='h6' noWrap>
                  {title}
                </Typography>
                {statusMessage}
                <div className={classes.grow}/>
                {buttons}
              </Toolbar>
            </AppBar>
    )
  }
}

export default withStyles(styles, { withTheme: true })(TitleBar)
