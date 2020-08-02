import AppBar from '@material-ui/core/AppBar'
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles'
import Toolbar from '@material-ui/core/Toolbar'
import React, { ReactNode } from 'react'
import { headerPageStyle } from '../styles/navigation_page'

export interface ClassType {
  /** root class */
  root: string
  /** top bar on page */
  appBar: string
}

export interface HeaderPageProps {
  /** divided page classes */
  classes: ClassType
  /** divided page children */
  children: {
    /** header content */
    headerContent: ReactNode
    /** main page content */
    pageContent: ReactNode
  }
}

/**
 * Renders a page split with a header
 * @param props
 * @constructor
 */
function HeaderPage (props: HeaderPageProps) {
  const { classes, children } = props
  return (
          <div className={classes.root}>
            <CssBaseline/>
            <AppBar
                    className={classes.appBar}
            >
              <Toolbar>
                {children.headerContent}
              </Toolbar>
            </AppBar>
            {children.pageContent}
          </div>
  )
}

/** export divided page */
export default withStyles(headerPageStyle, { withTheme: true })(HeaderPage)
