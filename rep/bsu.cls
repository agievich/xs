%%
%% This is file `bsu.cls'
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bsu}
              [1994/12/09 v1.2x
 Standard LaTeX document class]
\newcommand{\@ptsize}{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@openbib
\@openbibfalse
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\DeclareOption{9pt}{\renewcommand{\@ptsize}{9}}
\DeclareOption{10pt}{\renewcommand{\@ptsize}{10}}
\DeclareOption{11pt}{\renewcommand{\@ptsize}{11}}
\DeclareOption{12pt}{\renewcommand{\@ptsize}{12}}
\DeclareOption{14pt}{\renewcommand{\@ptsize}{14}}
\DeclareOption{17pt}{\renewcommand{\@ptsize}{17}}
\DeclareOption{20pt}{\renewcommand{\@ptsize}{20}}
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\DeclareOption{titlepage}{\@titlepagetrue}
\DeclareOption{notitlepage}{\@titlepagefalse}
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{\@openbibtrue}
\ExecuteOptions{letterpaper,10pt,oneside,onecolumn,final,openany}
\ProcessOptions
\input{size\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand{\baselinestretch}{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand{\topfraction}{.7}
\setcounter{bottomnumber}{1}
\renewcommand{\bottomfraction}{.3}
\setcounter{totalnumber}{3}
\renewcommand{\textfraction}{.2}
\renewcommand{\floatpagefraction}{.5}
\setcounter{dbltopnumber}{2}
\renewcommand{\dbltopfraction}{.7}
\renewcommand{\dblfloatpagefraction}{.5}

%%%% Registry Section %%%%%%%%%
\newcount{\c@registry}
\newcount{\c@pageregistry}
\@addtoreset{pageregistry}{page}
\def\theregistry{Уч.~\No~\arabic{pageregistry}с}
\newcommand{\registryformat}{}
\newenvironment{registrate}[1]
  {\setcounter{registry}{#1}}
  {\setcounter{pageregistry}{\c@registry}\setcounter{registry}{0}}

\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {\uppercase{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ %
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {\uppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\uppercase{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ %
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \def\@oddfoot{
      \ifnum\c@registry>0
        \setcounter{pageregistry}{\c@registry}
        \mbox{}\hfil\thepage\hfil\registryformat\theregistry
      \else
        \ifnum\c@pageregistry>0
           \mbox{}\hfil\thepage\hfil\registryformat\theregistry
        \else
          \mbox{}\hfil\thepage\hfil\mbox{}
        \fi
      \fi
    }
    \def\@evenfoot{
      \ifnum\c@registry>0
          \setcounter{pageregistry}{\c@registry}
          \mbox{}\hfil\thepage\hfil\registryformat\theregistry
      \else
        \ifnum\c@pageregistry>0
           \mbox{}\hfil\thepage\hfil\registryformat\theregistry
        \else
          \mbox{}\hfil\thepage\hfil\mbox{}
        \fi
      \fi
    }
    \let\@oddhead\@empty\let\@evenhead\@empty
    %\def\@evenhead{\thepage\hfil\slshape\leftmark}%
    %\def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
\if@titlepage
  \newcommand{\maketitle}{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \let\thanks\relax\let\maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}}
\else
\newcommand{\maketitle}{\par
  \begingroup
    \renewcommand{\thefootnote}{\fnsymbol{footnote}}%
    \def\@makefnmark{\hbox to\z@{$\m@th^{\@thefnmark}$\hss}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hbox to1.8em{\hss$\m@th^{\@thefnmark}$}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \let\thanks\relax
  \let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand*{\chaptermark}[1]{}
\setcounter{secnumdepth}{3}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\renewcommand{\thechapter}      {\arabic{chapter}}
\renewcommand{\thesection}      {\thechapter.\arabic{section}}
\renewcommand{\thesubsection}   {\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
\renewcommand{\theparagraph}{\thesubsubsection.\arabic{paragraph}}
\newcommand{\@chapapp}{}

\newcommand{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect
                                    \numberline{\thechapter~}#1} %<-- add point
                    \else
                      \addcontentsline{toc}{chapter}{\protect#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      % <--- delete \@afterheading there
                    \fi}
\def\@makechapterhead#1{%
  \vspace*{6\p@}%
  {%\parindent \z@ \reset@font %\centering
    \ifnum \c@secnumdepth >\m@ne
        \large\@chapapp{}\thechapter~ % <---- add point
    \fi
    #1\par\nobreak
    \vskip 2\p@
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   % <--- delete \@afterheading there
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{6\p@}%
  {%\parindent \z@ 
    \centering
    \reset@font
    \interlinepenalty\@M
    \large #1\par\nobreak
    \vskip 12\p@
  }}

%%%%%%%%%%%%%%% start   Points, points, points %%%%%%%%%%%%%%%%%%%%%%%%

\def\@sect#1#2#3#4#5#6[#7]#8
{
     \ifnum #2>\c@secnumdepth
     \def\@svsec{}\else
     \refstepcounter{#1}\edef\@svsec{\csname the#1\endcsname
                                  % <-- add point there ( numbering )
     \hskip 1em }\fi
     \@tempskipa #5\relax
      \ifdim \@tempskipa>\z@
        \begingroup
          \hyphenpenalty=10000 % <------ add penalty
          \raggedright         % <------ add right alignment
          #6\relax
          \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M #8\par}%
        \endgroup
       \csname #1mark\endcsname{#7}\addcontentsline
         {toc}{#1}{\ifnum #2>\c@secnumdepth \else
                      \protect\numberline{\csname the#1\endcsname
                                 % <-- add point there ( contents line )
                      }\fi
                    #7}\else
        \def\@svsechd{#6\hskip #3\@svsec #8\csname #1mark\endcsname
                      {#7}\addcontentsline
                           {toc}{#1}{\ifnum #2>\c@secnumdepth \else
                           \protect\numberline{\csname the#1\endcsname
                                 % <-- add point there ( contents line )
                           }\fi
                       #7}}\fi
     \@xsect{#5}
}

\def\@ssect#1#2#3#4#5
{
  \@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
     \begingroup
        \hyphenpenalty=10000 % <------ add penalty
        \raggedright         % <------ add right alignment
        #4
        \@hangfrom{\hskip #1}{\interlinepenalty \@M #5\par}
     \endgroup
  \else
    \def\@svsechd{#4\hskip #1\relax #5}
  \fi
  \@xsect{#3}
}


\renewcommand{\@begintheorem}[2]
{\begin{trivlist}\sl\item[\hspace{\labelsep}{\bf #1\ #2.}]} %<- add point

\renewcommand{\@endtheorem}{\end{trivlist}}

%%%%%%%%%%%%%%%%%  end  Points, points, points %%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Заголовки разделов ...
\newcommand{\section}{\@startsection{section}{1}{16pt}%
                                   {3.5ex \@plus 1ex \@minus .2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\reset@font\large}}

%%%%% Заголовки подразделов ... печатают начиная с абзаца
\newcommand{\subsection}{\@startsection{subsection}{2}{16pt}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\normalsize}}

%%%%% Заголовки пунктов ... печатают в разрядку или с использованием
%%%%% шрифтового выделения с абзаца в подбор к тексту
\newcommand{\subsubsection}{\@startsection{subsubsection}{3}{16pt}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\normalsize}}

%%%%% Заголовки пунктов ... печатают в разрядку или с использованием
%%%%% шрифтового выделения с абзаца в подбор к тексту
\newcommand{\paragraph}{\@startsection{paragraph}{4}{16pt}%
                                     {0.75ex\@plus 1ex \@minus .2ex}%
                                     {-1em}%
                                     {\reset@font\normalsize\bfseries}}

\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength\leftmargin    {\leftmargini}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand{\theenumi}{\arabic{enumi}}
\renewcommand{\theenumii}{\arabic{enumii}}
\renewcommand{\theenumiii}{\alph{enumiii}}
\renewcommand{\theenumiv}{\roman{enumiv}}
\newcommand{\labelenumi}{\theenumi)}
\newcommand{\labelenumii}{(\theenumii)}
\newcommand{\labelenumiii}{(\theenumiii)}
\newcommand{\labelenumiv}{(\theenumiv)}
\renewcommand{\p@enumii}{\theenumi}
\renewcommand{\p@enumiii}{\theenumi(\theenumii)}
\renewcommand{\p@enumiv}{\p@enumiii\theenumiii}
\newcommand{\labelitemi}{$\m@th\bullet$}
\newcommand{\labelitemii}{\normalfont\bfseries --}
\newcommand{\labelitemiii}{$\m@th\ast$}
\newcommand{\labelitemiv}{$\m@th\cdot$}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand{\descriptionlabel}[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\if@titlepage
  \newenvironment{abstract}{%
      \titlepage
      \null\vfil
      \begin{center}
        \bfseries \abstractname
      \end{center}}
     {\par\vfil\null\endtitlepage}
\else
  \newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi
\newenvironment{verse}
               {\let\\=\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item[]}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item[]}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item[]}
               {\endlist}
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \if@compatibility
        \setcounter{page}{0}
      \else
        \setcounter{page}{1}%
      \fi}%
    {\if@restonecol\twocolumn \else \newpage \fi
      \setcounter{page}{1}%
    }

%%%%%%%%%%%%% Appendix %%%%%%%%%%%%%%%%%%%

\newcounter{appendix}
\renewcommand{\theappendix}{\arabic{appendix}} %<- сквозная нумерация

\def\appendix#1
{
  \refstepcounter{appendix}
  \addcontentsline{toc}{chapter}%
  {\protect\appendixname~\numberline{\theappendix.} #1}
  \clearpage
  \vspace*{5\p@}%
  {
    \parindent \z@ \reset@font
    \raggedleft{\appendixname~\theappendix}
    \vskip 30\p@
  }
  {
    \parindent \z@ \reset@font\centering
    \huge #1
    \vskip 40\p@
  }
}
%%%%%%%%% End Appendix %%%%%%%%%%%%%%%%%%%

\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset{equation}{chapter}
\renewcommand{\theequation}{\thechapter.\arabic{equation}} 
\newcounter{figure}
\renewcommand{\thefigure}{\@arabic\c@figure} %<- сквозная нумерация
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure~--}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand{\thetable}{\@arabic\c@table} %<-- сквозная нумерация
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1 #2}%   % <------ :->.
  \ifdim \wd\@tempboxa >\hsize
    #1 #2\par               % <------ :->.
  \else
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

%%% Table Caption %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              Таблица (номер)
%%               Заголовок таблицы
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\tablecaption{\refstepcounter\@captype
\@dblarg{\@tablecaption\@captype}}

\long\def\@tablecaption#1[#2]#3{\par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \normalsize
    \@maketablecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}

\newlength\abovetablecaptionskip
\newlength\belowtablecaptionskip
\setlength\abovetablecaptionskip{4\p@}
\setlength\belowtablecaptionskip{10\p@}
\long\def\@maketablecaption#1#2{%
  \vskip\abovetablecaptionskip
  #1~--- #2\par
  \vskip\belowtablecaptionskip
}

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*{\cal}{\@fontswitch{\relax}{\mathcal}}
\DeclareRobustCommand*{\mit}{\@fontswitch{\relax}{\mathnormal}}
\newcommand{\@pnumwidth}{1.55em}
\newcommand{\@tocrmarg} {2.55em}
\newcommand{\@dotsep}{4.5}
\setcounter{tocdepth}{2}
\newcommand{\tableofcontents}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{\uppercase{\contentsname}}{\uppercase{\contentsname}}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*{\l@part}[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \setlength\@tempdima{3em}%
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hbox to\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}
    \endgroup
  \fi}
\newcommand*{\l@chapter}[2]{%
\raggedright
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
%      \hyphenpenalty=10000
      \parindent\z@ \rightskip\@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1
      \nobreak\hfill\nobreak
      \hbox to\@pnumwidth{\hss #2}
      \par
      \penalty\@highpenalty
    \endgroup
  \fi}

%\newskip\chaptocamount
%\chaptocamount=8pt
%\newbox\TestChapwd
%\setbox\TestChapwd=\hbox{Chapter 1.\hskip1em}%
%\def\numberline#1{\hbox to\@tempdima{\hfill#1\hskip1em}}%
%\def\chapterline#1{\hbox to\@tempdima{#1\hfill}}%
%\def\@tocline#1#2#3#4#5#6{\par
%  \addpenalty\@secpenalty\addvspace{#1}%
%  \begingroup
%\hyphenpenalty=10000
%\@tempdima#3\relax
%  \parindent\z@ \leftskip#2\relax \advance\leftskip by #3\relax
%  \rightskip\@pnumwidth \parfillskip-\@pnumwidth
%  {#4\leavevmode\hskip-\@tempdima #5\relax
%        \hfil\hbox to\@pnumwidth{\hss {\mdseries #6}}}\par\nobreak
%  \endgroup}
%\def\l@chapter{\@tocline\chaptocamount{0pt}{\wd\TestChapwd}{\upshape\normalfont}}
%
%
%
\newcommand*{\l@section}      {\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*{\l@subsection}   {\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*{\l@subsubsection}{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*{\l@paragraph}{\@dottedtocline{4}{7.0em}{4.1em}}

\newcommand{\listoffigures}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\uppercase{\listfigurename}}%
              {\uppercase{\listfigurename}}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*{\l@figure}{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand{\listoftables}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{\uppercase{\listtablename}}{\uppercase{\listtablename}}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\bibindent=1.5em
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\uppercase{\bibname}}{\uppercase{\bibname}}}%
      \addcontentsline{toc}{chapter}{\bibname}   % <--- add contents line
      \list{\@biblabel{\arabic{enumiv}}}%
           {\settowidth\labelwidth{\@biblabel{#1}}
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \if@openbib
              \advance\leftmargin\bibindent
              \itemindent -\bibindent
              \listparindent \itemindent
              \parsep \z@
            \fi
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand{\theenumiv}{\arabic{enumiv}}}%
      \if@openbib
        \renewcommand{\newblock}{\par}
      \else
        \renewcommand{\newblock}{\hskip .11em \@plus.33em \@minus.07em}%
      \fi
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.=\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

% Add new environment 'deflist'
\newenvironment{deflist}
     {\chapter*{\deflistname
        \@mkboth{\uppercase{\deflistname}}{\uppercase{\deflistname}}}%
        \addcontentsline{toc}{chapter}{\deflistname}}
     {}


\newcommand{\introductionname}{Введение}
\newcommand{\conclusionname}{Заключение}
\newcommand{\characteristicname}{Общая характеристика работы}

%%%%%%%%%%%% Chapters*

\newcommand{\introduction}
{\chapter*{\introductionname}
 \addcontentsline{toc}{chapter}{\introductionname}}

\newcommand{\conclusion}
{\chapter*{\conclusionname}
 \addcontentsline{toc}{chapter}{\conclusionname}}

\newcommand{\characteristic}
{\chapter*{\characteristicname}
 \addcontentsline{toc}{chapter}{\characteristicname}}

%%%% End Chapters*


\renewenvironment{enumerate}{%
  \ifnum \@enumdepth >3 \@toodeep\else
      \advance\@enumdepth \@ne
      \edef\@enumctr{enum\romannumeral\the\@enumdepth}\list
      {\csname label\@enumctr\endcsname}{
        \itemsep=6pt\parsep=0pt\parskip=0pt\topsep=3pt\partopsep=3pt
        \ifnum \@enumdepth =1
          \addtolength{\leftmargin}{-\parindent}
          \addtolength{\leftmargin}{-12pt}
        \fi
        \itemindent=\parindent
        \addtolength{\itemindent}{12pt}
        \labelsep=6pt
        \labelwidth=\parindent
        \listparindent=\parindent
        \addtolength{\labelwidth}{-\labelsep}
        \usecounter{\@enumctr}
        \def\makelabel##1{\hss\llap{\upshape##1}}}\fi}
{\endlist}

\let\endenumerate=\endlist % for efficiency

\renewenvironment{itemize}{%
  \ifnum\@itemdepth>3 \@toodeep
  \else \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \list{\csname\@itemitem\endcsname}{
        \itemsep=6pt\parsep=0pt\parskip=0pt\topsep=3pt\partopsep=3pt
        \addtolength{\leftmargin}{-\parindent}
        \addtolength{\leftmargin}{-12pt}
        \itemindent=\parindent
        \addtolength{\itemindent}{12pt}
        \labelsep=6pt
        \labelwidth=\parindent
        \listparindent=\parindent
        \addtolength{\labelwidth}{-\labelsep}
       \def\makelabel##1{\hss\llap{\upshape##1}}}%
  \fi
}{%
  \endlist
}
\let\enditemize=\endlist % for efficiency


\newcommand{\newblock}{}
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\uppercase{\indexname}}%
                        {\uppercase{\indexname}}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand{\@idxitem}  {\par\hangindent 40\p@}
\newcommand{\subitem}   {\par\hangindent 40\p@ \hspace*{20\p@}}
\newcommand{\subsubitem}{\par\hangindent 40\p@ \hspace*{30\p@}}
\newcommand{\indexspace}{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand{\footnoterule}{%
  \kern-3\p@
  \hrule width .4\columnwidth
  \kern 2.6\p@}
\@addtoreset{footnote}{page} %<-- reset with new page
\long\def\@makefntext#1{%
    \parindent 1em%
    \noindent
    \hbox to 1.8em{\hss\@makefnmark}#1}
\newcommand{\contentsname}{Содержание}
\newcommand{\listfigurename}{Список рисунков}
\newcommand{\listtablename}{Список таблиц}
\newcommand{\bibname}{Список использованных источников}
\newcommand{\refname}{Список использованных источников}
\newcommand{\deflistname}{Перечень условных обозначений и сокращений}
\newcommand{\indexname}{Предметный указатель}
\newcommand{\figurename}{Рисунок}
\newcommand{\tablename}{Таблица}
\newcommand{\partname}{Часть}
\newcommand{\chaptername}{}
\newcommand{\appendixname}{Приложение}
\newcommand{\abstractname}{Аннотация}
\newcommand{\today}{\number\day\space\ifcase\month\or
  января\or февраля\or марта\or апреля\or мая\or июня\or
  июля\or августа\or сентября\or октября\or ноября\or декабря\fi
  \space\number\year\space г.}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}      % Arabic page numbers
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi

%%%%%%%%%%%%%%%%% Дата подписания %%%%%%%%%%%%%%%%%%%%
\newcommand{\signdate}{\today}

\endinput
%%
%% End of file `bsu.cls'.
