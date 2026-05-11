Github pages blog

## Local development

This site is built with Jekyll and served by GitHub Pages. The repository pins
Ruby with `.ruby-version`; use `rbenv` so local commands run against that Ruby
instead of the system Ruby.

```sh
rbenv install 3.3.0
rbenv local 3.3.0
rbenv exec gem install bundler
rbenv exec bundle install
npm run serve
```

Useful commands:

```sh
npm run prepare:glucose
npm run build
npm run check
```

The glucose data prep script reads the local source files in
`/Users/jhuck/Documents/work/Glucose` by default and writes static JSON into
`assets/data`.
