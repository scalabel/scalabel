import React from 'react';


type Props = {
  views: Array<Object>
}

/**
 * Canvas Viewer
 */
class MainView extends React.Component<Props> {
  divRef: Object;

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor(props: Object) {
    super(props);
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  render() {
    let rectDiv;
    if (this.divRef) {
      rectDiv = this.divRef.getBoundingClientRect();
    }

    const {views} = this.props;
    let viewsWithProps = views;
    if (rectDiv) {
      viewsWithProps = React.Children.map(views, (view) => {
            if (rectDiv) {
              return React.cloneElement(view,
                  {height: rectDiv.height, width: rectDiv.width});
            } else {
              return React.cloneElement(view, {});
            }
          }
      );
    }

    return (
        <div ref={(element) => {
          if (element) {
            this.divRef = element;
          }
        }}
             style={{
               display: 'block', height: '100%', position: 'absolute',
               outline: 'none', width: '100%', background: '#222222',
             }}>
          {viewsWithProps}
        </div>
    );
  }
}

export default MainView;
