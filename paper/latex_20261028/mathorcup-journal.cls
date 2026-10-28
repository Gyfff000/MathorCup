%% MathorCup adaptation of the cjc class framework, 2026-10-28.
%% Original work: cjc, Copyright (C) 2019 Zeping Lee.
%% Source: https://github.com/latexstudio/chinesejournal
%% Distributed under LPPL 1.3c or later; see LICENSE and cjc.dtx.
%% The upstream class was changed and renamed for current ctex and contest rules.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mathorcup-journal}[2026/10/28 Contest adaptation of cjc]
\LoadClass[UTF8,a4paper,twocolumn,zihao=5,linespread=1.08]{ctexart}
\RequirePackage{amsmath,amssymb,geometry,graphicx,fancyhdr,caption,url}
\geometry{paper=a4paper,vmargin=2.6cm,hmargin=2cm,columnsep=0.75cm}
\ctexset{
  section={format=\bfseries\zihao{4},beforeskip=8bp,afterskip=8bp},
  subsection={format=\bfseries\zihao{5},beforeskip=2.625bp,afterskip=2.625bp},
  subsubsection={format=\zihao{5},beforeskip=0bp,afterskip=0bp}
}
\DeclareCaptionLabelSeparator{zhspace}{\hspace{\ccwd}}
\captionsetup{justification=centerlast,font=small,labelsep=zhspace,skip=6bp,
  figureposition=bottom,tableposition=top}
\DeclareCaptionFont{heiti}{\heiti}
\captionsetup[table]{font={heiti,small}}
\sloppy
\raggedbottom
\setlength{\parskip}{0pt}
\urlstyle{same}
\endinput
