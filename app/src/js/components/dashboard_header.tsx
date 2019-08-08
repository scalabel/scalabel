import { withStyles } from '@material-ui/core'
import IconButton from '@material-ui/core/IconButton'
import SvgIcon from '@material-ui/core/SvgIcon'
import Typography from '@material-ui/core/Typography'
import React from 'react'
import { logout } from '../common/service'
import { dashboardHeaderStyles } from '../styles/dashboard'

interface DashboardHeaderClassType {
  /** title styling */
  title: string
}

/**
 * renders dashboard header for worker and admin
 * @param props
 * @constructor
 */
function DashboardHeader (props: {
  /** dashboard header classes */
  classes: DashboardHeaderClassType
  /** if this is the admin handler */
  admin?: boolean
}) {
  const { classes, admin } = props
  return (
          <React.Fragment>
            <Typography
                    component='h1'
                    variant='h6'
                    color='inherit'
                    noWrap
                    className={classes.title}
            >{admin ? 'Scalabel Admin Dashboard' :
              'Scalabel Worker Dashboard'}
            </Typography>
            <IconButton
                    onClick={logout}
            >
              <SvgIcon>
                <path d='M10.09 15.59L11.5 17l5-5-5-5-1.41 1.41L12.67
                    11H3v2h9.67l-2.58 2.59zM19 3H5c-1.11 0-2 .9-2
                    2v4h2V5h14v14H5v-4H3v4c0 1.1.89 2 2 2h14c1.1 0 2-.9
                    2-2V5c0-1.1-.9-2-2-2z' fill='#ffffff'/>
              </SvgIcon>
            </IconButton>
          </React.Fragment>
  )
}

export default withStyles(dashboardHeaderStyles)(DashboardHeader)
