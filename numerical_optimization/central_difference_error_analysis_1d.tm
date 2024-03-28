<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Central Difference Error Analysis \U 1D Case>>

  Let\ 

  \;

  Taylor expansion:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x+\<varepsilon\>|)>>|<cell|=>|<cell|f<around*|(|x|)>+f<rsup|<rprime|'>><around*|(|x|)>\<varepsilon\>+<frac|f<rsup|<rprime|''>><around*|(|x|)>|2!>\<varepsilon\><rsup|2>+<frac|f<rprime|'''><around*|(|x|)>|3!>\<varepsilon\><rsup|3>+\<cdots\>>>>>
  </eqnarray*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x-\<varepsilon\>|)>>|<cell|=>|<cell|f<around*|(|x|)>-f<rsup|<rprime|'>><around*|(|x|)>\<varepsilon\>+<frac|f<rsup|<rprime|''>><around*|(|x|)>|2!>\<varepsilon\><rsup|2>-<frac|f<rprime|'''><around*|(|x|)>|3!>\<varepsilon\><rsup|3>+\<cdots\>>>>>
  </eqnarray*>

  Note how the signs alternate.

  Subtract from the first expression, the second expression. Odd terms do not
  get cancelled out, but even terms get doubled:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x+\<varepsilon\>|)>-f<around*|(|x-\<varepsilon\>|)>>|<cell|=>|<cell|2f<rsup|<rprime|'>><around*|(|x|)>\<varepsilon\>+2<frac|f<rprime|'''><around*|(|x|)>|3!>\<varepsilon\><rsup|3>+2<frac|f<rsup|<around*|(|5|)>><around*|(|x|)>|5!>\<varepsilon\><rsup|5>+\<cdots\>+>>|<row|<cell|f<around*|(|x+\<varepsilon\>|)>-f<around*|(|x-\<varepsilon\>|)>>|<cell|=>|<cell|2f<rsup|<rprime|'>><around*|(|x|)>\<varepsilon\>+O<around*|(|\<varepsilon\><rsup|3>|)><space|1em><around*|(|<text|as
    >\<varepsilon\>\<rightarrow\>0|)>>>|<row|<cell|f<rprime|'><around*|(|x|)>>|<cell|=>|<cell|<frac|f<around*|(|x+\<varepsilon\>|)>-f<around*|(|x-\<varepsilon\>|)>|\<varepsilon\>>+O<around*|(|\<varepsilon\><rsup|2>|)>>>>>
  </eqnarray*>

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-screen-margin|false>
  </collection>
</initial>