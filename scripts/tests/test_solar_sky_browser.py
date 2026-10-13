"""Opt-in real-browser projection/occlusion contract, run by gallery CI."""
import base64
from io import BytesIO
import os
import shutil

from PIL import Image
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('SSATK_BROWSER_TESTS') != '1',
                                reason='Set SSATK_BROWSER_TESTS=1 for headless Chrome checks')


def test_sky_rotates_resizes_and_stays_behind_opaque_geometry(tmp_path):
    """C/R1: two opposite known rays; pixel checks distinguish sky from foreground.

    The fixture is analytic, not a real catalogue. Blue star faces +z camera,
    red star faces -z camera; the green square must occlude either center star.
    """
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from ssatk.plots._solar_scene import SolarFigure, sky_source
    options=webdriver.ChromeOptions()
    chrome = os.environ.get("SSATK_CHROME_BINARY") or shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        options.binary_location = chrome
    for arg in ('--headless=new','--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'):
        options.add_argument(arg)
    options.set_capability('goog:loggingPrefs',{'browser':'ALL'})
    driver=webdriver.Chrome(options=options)
    camera=dict(eye=dict(x=0,y=0,z=2),up=dict(x=0,y=1,z=0))
    sky=dict(directions=[[0,0,-1],[0,0,1]],sizes=[8,8],colors=['rgb(0,0,255)','rgb(255,0,0)'])
    fig=SolarFigure()
    fig.add_mesh3d(x=[-.4,.4,.4,-.4],y=[-.4,-.4,.4,.4],z=[0,0,0,0],
                  i=[0,0],j=[1,2],k=[2,3],color='#00ff00',
                  lighting=dict(ambient=1,diffuse=0,specular=0,fresnel=0,roughness=1))
    axis=dict(range=[-1,1],visible=False)
    fig.update_layout(scene=dict(xaxis=axis,yaxis=axis,zaxis=axis,camera=camera,
                                 aspectmode='cube',bgcolor='rgba(0,0,0,0)'),
                      margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='black',
                      meta=dict(solar_sky=sky))
    fig.add_layout_image(source=sky_source(sky,camera,800,600),xref='paper',yref='paper',
                         x=0,y=1,sizex=1,sizey=1,sizing='stretch',layer='below')
    fig.frames = [dict(name='front', layout=dict(scene=dict(camera=camera)))]
    path=tmp_path/'sky.html'
    fig.write_html(path, auto_play=False)
    script="return document.querySelector('.plotly-graph-div').layout.images[0].source"
    def pixel(source):
        image=Image.open(BytesIO(base64.b64decode(source.split(',')[1]))).convert('RGB')
        return image.getpixel((image.width//2,image.height//2))
    try:
        driver.set_window_size(1000,800)
        driver.get(path.as_uri())
        wait=WebDriverWait(driver,45)
        wait.until(lambda d:d.execute_script("return !!document.querySelector('.plotly-graph-div')?.layout?.images?.[0]?.source?.startsWith('data:image/png')"))
        source=driver.execute_script(script)
        assert pixel(source)==(0,0,255)
        # Wait for Plotly's actual WebGL render, then inspect the viewport center.
        def center_is_green(d):
            png=Image.open(BytesIO(d.find_element('css selector','.plotly-graph-div').screenshot_as_png)).convert('RGB')
            r,g,b=png.getpixel((png.width//2,png.height//2))
            return g>200 and r<30 and b<30
        wait.until(center_is_green)
        driver.execute_script("Plotly.relayout(document.querySelector('.plotly-graph-div'), {'scene.camera': {eye:{x:0,y:0,z:-2},up:{x:0,y:1,z:0}}})")
        wait.until(lambda d:d.execute_script(script)!=source)
        opposite=driver.execute_script(script)
        assert pixel(opposite)==(255,0,0)
        wait.until(center_is_green)
        driver.set_window_size(1300,700)
        wait.until(lambda d:d.execute_script(script)!=opposite)
        assert pixel(driver.execute_script(script))==(255,0,0)
        # A scripted animation must refresh the sky too, without a mouse event.
        driver.execute_script("Plotly.animate(document.querySelector('.plotly-graph-div'), ['front'], {mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}})")
        wait.until(lambda d:pixel(d.execute_script(script))==(0,0,255))
        wait.until(center_is_green)
        errors=[e['message'] for e in driver.get_log('browser') if e['level']=='SEVERE']
        assert errors==[]
    finally:
        driver.quit()
