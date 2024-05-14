#!/usr/bin/env python3
# Developed by jxLiang
class colorizePrint(object):
    def __init__(self, initstr=None) -> None:
        self.foregroundColorMapping = {
            "Default": "37",
            "Black": "30",
            "Red": "31",
            "Green": "32",
            "Yellow": "33",
            "Blue": "34",
            "Magenta": "35",  # 品红色
            "Cyan": "36",  # 青色
            "White": "37",
        }

        self.backgroundColorMapping = {
            "Default": "38",
            "Black": "40",
            "Red": "41",
            "Green": "42",
            "Yellow": "43",
            "Blue": "44",
            "Magenta": "45",  # 品红色
            "Cyan": "46",  # 青色
            "White": "47",
        }

        self.fontsettingMapping = {
            "Default": "0",
            "Highlight": "1",  # 高亮
            "Halfhighlight": "2",  # 一半高亮
            "Italic": "3",  # 斜体
            "Underline": "4",  # 下划线
            "Twinkle": "5",  # 闪烁
            "Reverse": "7",  # 反显
            "Disapper": "8",
        }

        self.defaultSetting = "\033[0m"

        # self.fgc = "Default"
        # self.bgc = "Default"
        # self.fontsetting= "Default"
        if initstr != None:
            self.greenp(initstr)

    def outprintMapping(
        self,
        foregroundColor="Default",
        backgroundColor="Default",
        fontsetting="Default",
    ):
        foregroundColor = self.foregroundColorMapping[foregroundColor]
        backgroundColor = self.backgroundColorMapping[backgroundColor]
        fontsetting = self.fontsettingMapping[fontsetting]
        setting = (
            "\033[" + fontsetting + ";" + foregroundColor + ";" + backgroundColor + "m"
        )

        # print("foregroundColor", foregroundColor)
        # print("backgroundColor", backgroundColor)
        # print("fontsetting", fontsetting)
        # print("setting", setting)

        return setting

    def cprint(
        self,
        *values: object,
        fgc="Default",  # foregroundColor
        bgc="Default",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        setting = self.outprintMapping(
            foregroundColor=fgc, backgroundColor=bgc, fontsetting=fontsetting
        )
        print(setting, sep="", end="")
        print(*values, sep=sep, end="")
        print(self.defaultSetting, sep="", end=end)

    def __call__(  # The param is according to the selfparam
        self,
        *values: object,
        fgc="Default",  # foregroundColor
        bgc="Default",  # backgroundColor
        fontsetting="Default"  # fontsetting
    ):
        # fgc = self.fgc
        # bgc = self.bgc
        # fontsetting = self.fontsetting
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting)

    # def setSelfParam(
    #         self,
    #         fgc = "Default",
    #         bgc = "Default",
    #         fontsetting = "Default"
    #         ):
    #     self.fgc=fgc
    #     self.bgc=bgc
    #     self.fontsetting=fontsetting

    def redp(  # fgc == red   -->  redprint
        self,
        *values: object,
        fgc="Red",  # foregroundColor
        bgc="Default",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting,sep=sep,end=end)

    def yellowp(  # fgc == yellow   -->  yellowprint
        self,
        *values: object,
        fgc="Yellow",  # foregroundColor
        bgc="Default",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting,sep=sep,end=end)

    def greenp(  # fgc == green   -->  greenprint
        self,
        *values: object,
        fgc="Green",  # foregroundColor
        bgc="Default",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting,sep=sep,end=end)

    def bluep(  # fgc == blue   -->  blueprint
        self,
        *values: object,
        fgc="Blue",  # foregroundColor
        bgc="Default",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting,sep=sep,end=end)

    
    def warning(  # fgc == Black  bgc==Yellow  
        self,
        *values: object,
        fgc="Black",  # foregroundColor
        bgc="Yellow",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        self.cprint(" Warning: ", fgc=fgc, bgc=bgc, fontsetting="Highlight",end="")
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting,sep=sep,end=end)

    def error(  # fgc == Black  bgc==Yellow  
        self,
        *values: object,
        fgc="Black",  # foregroundColor
        bgc="Red",  # backgroundColor
        fontsetting="Default",  # fontsetting
        sep=" ",
        end="\n"
    ):
        self.cprint(" Error: ", fgc=fgc, bgc=bgc, fontsetting="Highlight",end="")
        self.cprint(*values, fgc=fgc, bgc=bgc, fontsetting=fontsetting,sep=sep,end=end)

    def greenhp(    # fgc == green  fontsetting == highlight -->  greenhighlightprint
                self,
                *values: object,
                fgc="Green",  # foregroundColor
                bgc="Default",  # backgroundColor
                fontsetting="Highlight",  # fontsetting
                sep=" ",
                end="\n"
    ):
        self.greenp(*values,fgc=fgc,bgc=bgc,fontsetting=fontsetting,sep=sep,end=end)

    def yellowhp(    # fgc == yellow  fontsetting == highlight -->  yellowhighlightprint
                self,
                *values: object,
                fgc="Yellow",  # foregroundColor
                bgc="Default",  # backgroundColor
                fontsetting="Highlight",  # fontsetting
                sep=" ",
                end="\n"
    ):
        self.yellowp(*values,fgc=fgc,bgc=bgc,fontsetting=fontsetting,sep=sep,end=end)

    def bluehp(    # fgc == blue  fontsetting == highlight -->  bluehighlightprint
                self,
                *values: object,
                fgc="Blue",  # foregroundColor
                bgc="Default",  # backgroundColor
                fontsetting="Highlight",  # fontsetting
                sep=" ",
                end="\n"
    ):
        self.bluep(*values,fgc=fgc,bgc=bgc,fontsetting=fontsetting,sep=sep,end=end)

    def redhp(    # fgc == red  fontsetting == highlight -->  bluehighlightprint
                self,
                *values: object,
                fgc="Red",  # foregroundColor
                bgc="Default",  # backgroundColor
                fontsetting="Highlight",  # fontsetting
                sep=" ",
                end="\n"
    ):
        self.redp(*values,fgc=fgc,bgc=bgc,fontsetting=fontsetting,sep=sep,end=end)